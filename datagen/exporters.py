"""Dataset exporters.

Formats:
  jsonl       raw Record rows, everything including scores and provenance
  alpaca      {instruction, input, output}          — classic SFT
  sharegpt    {conversations:[{from,value},...]}    — axolotl / LLaMA-Factory
  chatml      {messages:[{role,content},...]}       — OpenAI-style SFT
  rag_chunks  the chunks themselves, for retrieval rather than fine-tuning
  kalam_kb    writes into server/pcai/learned.json so the existing Kalam PCAI
              assistant picks the material up on its next "Train"

Everything is split train/eval by SOURCE DOCUMENT, not by row: rows generated
from the same chunk are near-duplicates, and letting them straddle the split is
how you get an eval score that means nothing.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .models import Chunk, Record
from .util import get_logger, now_iso

log = get_logger("exporters")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    log.info("wrote %-46s %d rows", path.name, n)
    return n


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


# ---------------------------------------------------------------------------
# Row shapes
# ---------------------------------------------------------------------------


def to_alpaca(rec: Record, cite: bool = True) -> dict:
    out = rec.output
    if cite and rec.source_url:
        out = f"{out}\n\nSource: {rec.source_title} <{rec.source_url}>"
    return {"instruction": rec.instruction, "input": rec.input, "output": out}


def to_sharegpt(rec: Record) -> dict:
    return {
        "conversations": [
            {"from": "human", "value": rec.instruction},
            {"from": "gpt", "value": rec.output},
        ],
        "source": rec.source_url,
    }


def to_chatml(rec: Record, system: str = "") -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages += [
        {"role": "user", "content": rec.instruction},
        {"role": "assistant", "content": rec.output},
    ]
    return {"messages": messages, "metadata": {"source": rec.source_url, "score": rec.score}}


def to_rag_chunk(chunk: Chunk) -> dict:
    """Shape mirrors the Chunk interface in server/pcai/store.ts."""
    return {
        "id": chunk.id,
        "title": chunk.title,
        "url": chunk.url,
        "text": chunk.text,
        "tokens": chunk.tokens,
        **({"embedding": chunk.embedding} if chunk.embedding else {}),
    }


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_by_document(records: list[Record], train_ratio: float, seed: int) -> tuple[list[Record], list[Record]]:
    """Group-aware split: all rows from one source URL land on the same side.

    Groups are packed largest-first, which matters when the corpus has only a
    handful of documents: filling in random order can overshoot on the first
    group and dump nearly everything into eval. With few documents the split
    can only ever approximate `train_ratio` — it lands on the closest option
    rather than the worst one.
    """
    groups: dict[str, list[Record]] = {}
    for rec in records:
        groups.setdefault(rec.source_url or rec.chunk_id, []).append(rec)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)          # reproducible tie-breaking …
    keys.sort(key=lambda k: len(groups[k]), reverse=True)   # … largest first

    target = len(records) * train_ratio
    train: list[Record] = []
    evalset: list[Record] = []
    for key in keys:
        group = groups[key]
        # Take the group into train while it still fits, and always take the
        # first one so train is never empty.
        if not train or len(train) + len(group) <= target:
            train.extend(group)
        else:
            evalset.extend(group)

    # Guarantee a non-empty eval set by moving the SMALLEST group across —
    # that is the least damaging way to carve one out.
    if not evalset and len(keys) > 1:
        smallest = min(keys, key=lambda k: len(groups[k]))
        moved = {id(r) for r in groups[smallest]}
        evalset = groups[smallest]
        train = [r for r in train if id(r) not in moved]

    log.info(
        "split: %d train / %d eval across %d source documents (target %.0f train)",
        len(train), len(evalset), len(keys), target,
    )
    if len(keys) < 5:
        log.info(
            "  only %d source documents — a document-level split is coarse; "
            "add more sources for a meaningful eval set", len(keys),
        )
    return train, evalset


# ---------------------------------------------------------------------------
# Glossary
#
# Unlike every other format, a glossary is not one-row-per-record: the same term
# gets defined in a dozen chunks and the reader wants ONE entry with the best
# definition and every place it is discussed. So entries are merged by term.
# ---------------------------------------------------------------------------

_TERM_KEY = re.compile(r"[^a-z0-9]+")


def _term_key(term: str) -> str:
    """Merge key: case, punctuation and plural-insensitive."""
    key = _TERM_KEY.sub(" ", term.lower()).strip()
    return key[:-1] if len(key) > 3 and key.endswith("s") and not key.endswith("ss") else key


def build_glossary(records: list[Record]) -> list[dict]:
    """Merge glossary records into one entry per term.

    The winning definition is the highest-scored one; ties break toward the
    longer (more informative) text. Every source that mentioned the term is
    kept, so an entry cites all of them.
    """
    merged: dict[str, dict] = {}

    for rec in records:
        if rec.kind != "glossary" or rec.quarantined:
            continue
        term = (rec.meta or {}).get("term") or rec.instruction.removeprefix("What is ").rstrip("?")
        term = term.strip()
        if not term:
            continue

        key = _term_key(term)
        entry = merged.setdefault(
            key,
            {"term": term, "definition": "", "score": -1.0, "aliases": set(),
             "sources": {}, "mentions": 0, "generator": ""},
        )
        entry["mentions"] += 1
        entry["aliases"].update((rec.meta or {}).get("aliases") or [])

        score = rec.score if rec.score is not None else 0.0
        better = score > entry["score"] or (
            score == entry["score"] and len(rec.output) > len(entry["definition"])
        )
        if better:
            entry.update(
                {"definition": rec.output, "score": score, "term": term,
                 "generator": rec.generator}
            )
        if rec.source_url:
            entry["sources"][rec.source_url] = rec.source_title or rec.source_url

    out = []
    for entry in merged.values():
        aliases = sorted(
            a for a in entry["aliases"]
            if a and _term_key(a) != _term_key(entry["term"])
        )
        out.append({
            "term": entry["term"],
            "definition": entry["definition"],
            "aliases": aliases,
            "sources": [{"title": t, "url": u} for u, t in entry["sources"].items()],
            "mentions": entry["mentions"],
            "score": round(entry["score"], 3) if entry["score"] >= 0 else None,
            "generator": entry["generator"],
        })

    out.sort(key=lambda e: e["term"].lower())
    log.info("glossary: %d unique terms from %d records",
             len(out), sum(1 for r in records if r.kind == "glossary"))
    return out


def glossary_markdown(entries: list[dict], cfg: Config) -> str:
    """Render the glossary as a readable document, grouped A-Z."""
    groups: dict[str, list[dict]] = {}
    for entry in entries:
        first = entry["term"][0].upper()
        groups.setdefault(first if first.isalpha() else "#", []).append(entry)

    letters = sorted(groups, key=lambda c: (c == "#", c))

    lines = [
        f"# Glossary — {cfg.description or cfg.name}",
        "",
        f"{len(entries)} terms, generated {now_iso()} by the local model "
        f"`{cfg.llm.model}`. Every definition is drawn from the sources cited "
        "beneath it — nothing here is the model's own knowledge.",
        "",
        "  ".join(f"[{c}](#{c.lower() if c.isalpha() else 'other'})" for c in letters),
        "",
    ]

    for letter in letters:
        lines += [f"## {letter}" if letter.isalpha() else "## Other", ""]
        for entry in groups[letter]:
            lines.append(f"### {entry['term']}")
            lines.append("")
            lines.append(entry["definition"])
            lines.append("")
            if entry["aliases"]:
                lines.append(f"*Also known as: {', '.join(entry['aliases'])}*")
                lines.append("")
            if entry["sources"]:
                cites = ", ".join(
                    f"[{s['title'][:60]}]({s['url']})" for s in entry["sources"][:4]
                )
                extra = f" +{len(entry['sources']) - 4} more" if len(entry["sources"]) > 4 else ""
                lines.append(f"<sub>Sources: {cites}{extra}</sub>")
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Kalam PCAI bridge
# ---------------------------------------------------------------------------


def export_kalam_kb(chunks: list[Chunk], records: list[Record], path: Path, dataset_name: str) -> int:
    """Append into `server/pcai/learned.json` (the LearnedDoc[] the TS side reads).

    Existing entries from this dataset are replaced; anything the user added
    through the Kalam UI is left untouched.
    """
    tag = f"datagen://{dataset_name}/"
    existing: list[dict] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = [d for d in loaded if not str(d.get("url", "")).startswith(tag)]
        except (json.JSONDecodeError, OSError) as e:
            log.warning("could not read %s (%s) — starting a fresh list", path, e)

    docs = [
        {
            "title": c.title,
            "url": f"{tag}chunk/{c.id}",
            "text": c.text,
            "kind": "runbook" if c.kind == "runbook" else "note",
            "addedAt": now_iso(),
        }
        for c in chunks
    ]

    # Q&A pairs go in as their own docs — they retrieve well for error queries.
    for rec in records:
        if rec.kind != "troubleshooting":
            continue
        docs.append(
            {
                "title": f"Case: {rec.instruction[:80]}",
                "url": f"{tag}case/{rec.id}",
                "text": f"Problem: {rec.instruction}\n\nResolution: {rec.output}\n\n"
                        f"Source: {rec.source_title} <{rec.source_url}>",
                "kind": "case",
                "addedAt": now_iso(),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing + docs, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %d docs into the Kalam KB at %s (kept %d existing)",
             len(docs), path, len(existing))
    return len(docs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def export_all(
    cfg: Config,
    records: list[Record] | None = None,
    chunks: list[Chunk] | None = None,
) -> dict[str, Any]:
    """Export everything the config asks for. Reads from data/*.jsonl when the
    caller does not pass records/chunks, so `datagen export` works standalone."""
    out_dir = cfg.export_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if records is None:
        rows = _read_jsonl(cfg.records_path)
        records = [Record(**{k: v for k, v in r.items() if k in Record.__annotations__}) for r in rows]
    records = [r for r in records if not r.quarantined]

    if chunks is None:
        rows = _read_jsonl(cfg.chunks_path)
        chunks = [Chunk(**{k: v for k, v in r.items() if k in Chunk.__annotations__}) for r in rows]

    if not records and not chunks:
        log.warning("nothing to export — run `datagen build` first")
        return {"records": 0, "chunks": 0, "files": []}

    train, evalset = split_by_document(records, cfg.export.train_split, cfg.export.shuffle_seed)
    formats = set(cfg.export.formats)
    written: list[str] = []
    system = f"You are an expert assistant for {cfg.description or cfg.name}."

    def emit(name: str, rows: Iterable[dict]) -> None:
        _write_jsonl(out_dir / name, rows)
        written.append(name)

    if "jsonl" in formats:
        emit("dataset.jsonl", (r.to_dict() for r in records))
        emit("train.jsonl", (r.to_dict(include_context=False) for r in train))
        emit("eval.jsonl", (r.to_dict(include_context=False) for r in evalset))

    if "alpaca" in formats:
        emit("train.alpaca.jsonl", (to_alpaca(r, cfg.generation.include_citations) for r in train))
        emit("eval.alpaca.jsonl", (to_alpaca(r, cfg.generation.include_citations) for r in evalset))

    if "sharegpt" in formats:
        emit("train.sharegpt.jsonl", (to_sharegpt(r) for r in train))
        emit("eval.sharegpt.jsonl", (to_sharegpt(r) for r in evalset))

    if "chatml" in formats:
        emit("train.chatml.jsonl", (to_chatml(r, system) for r in train))
        emit("eval.chatml.jsonl", (to_chatml(r, system) for r in evalset))

    if "rag_chunks" in formats and chunks:
        emit("rag_chunks.jsonl", (to_rag_chunk(c) for c in chunks))

    glossary: list[dict] = []
    if "glossary" in formats:
        glossary = build_glossary(records)
        if glossary:
            emit("glossary.jsonl", iter(glossary))
            doc = out_dir / "GLOSSARY.md"
            doc.write_text(glossary_markdown(glossary, cfg), encoding="utf-8")
            log.info("wrote %-46s %d terms", doc.name, len(glossary))
            written.append(doc.name)
        else:
            log.info("glossary: no terms found — is 'glossary' in generation.kinds?")

    kalam_docs = 0
    if cfg.export.kalam_kb_path:
        kalam_docs = export_kalam_kb(
            chunks, records, cfg.resolve(cfg.export.kalam_kb_path), cfg.name
        )
        written.append(cfg.export.kalam_kb_path)

    manifest = {
        "dataset": cfg.name,
        "generated_at": now_iso(),
        "generated_by": f"local model {cfg.llm.model} via {cfg.llm.provider} ({cfg.llm.base_url})",
        "records": len(records),
        "train": len(train),
        "eval": len(evalset),
        "chunks": len(chunks),
        "glossary_terms": len(glossary),
        "kalam_kb_docs": kalam_docs,
        "kinds": _tally(r.kind for r in records),
        "sources": _tally(t for r in records for t in r.tags if t.startswith("source:")),
        "mean_score": round(
            sum(r.score or 0 for r in records) / len(records), 3
        ) if records else 0,
        "files": written,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("export complete -> %s", out_dir)
    return manifest


def _tally(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
