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
    """Group-aware split: all rows from one source URL land on the same side."""
    groups: dict[str, list[Record]] = {}
    for rec in records:
        groups.setdefault(rec.source_url or rec.chunk_id, []).append(rec)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    target = int(len(records) * train_ratio)
    train: list[Record] = []
    evalset: list[Record] = []
    for key in keys:
        (train if len(train) < target else evalset).extend(groups[key])

    # Guarantee a non-empty eval set when there is enough material for one.
    if not evalset and len(keys) > 1:
        moved = groups[keys[-1]]
        train = [r for r in train if r not in moved]
        evalset = moved

    log.info(
        "split: %d train / %d eval across %d source documents",
        len(train), len(evalset), len(keys),
    )
    return train, evalset


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
