"""Dataset analysis: is this actually trainable?

Counts alone do not tell you whether a dataset will fine-tune well. What
matters, roughly in order of how often it silently ruins a run:

  1. truncation      rows longer than max_seq_len get cut mid-answer, so the
                     model learns to stop early
  2. leakage         the same question in train and eval makes your eval loss
                     a lie
  3. duplication     surviving near-duplicates get over-weighted
  4. imbalance       one kind or one source dominating teaches format, not
                     knowledge
  5. volume          too few rows to move the weights at all
  6. degenerate rows empty, one-word or truncated-looking answers

`datagen analyze` reports all of it and prints actionable warnings; the same
report becomes DATASET_CARD.md next to the exports.

Everything here is stdlib and works on the exported JSONL, so it is also a
sanity check on data you built weeks ago.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Config
from .dedupe import _SimhashIndex, normalize_for_hash, simhash
from .models import Record
from .util import estimate_tokens, get_logger, now_iso, truncate

log = get_logger("analyze")

# Sizes below which a supervised fine-tune is unlikely to do much. Rough
# community consensus, not a law — a narrow domain can work with fewer.
MIN_USEFUL_RECORDS = 500
MIN_VIABLE_RECORDS = 100


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile. `values` must be sorted."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return float(values[lo] + (values[hi] - values[lo]) * (k - lo))


@dataclass
class LengthStats:
    count: int = 0
    mean: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    maximum: float = 0.0
    minimum: float = 0.0

    @staticmethod
    def of(values: Iterable[float]) -> "LengthStats":
        vals = sorted(float(v) for v in values)
        if not vals:
            return LengthStats()
        return LengthStats(
            count=len(vals),
            mean=round(sum(vals) / len(vals), 1),
            p50=round(percentile(vals, 0.50), 1),
            p90=round(percentile(vals, 0.90), 1),
            p95=round(percentile(vals, 0.95), 1),
            p99=round(percentile(vals, 0.99), 1),
            maximum=round(vals[-1], 1),
            minimum=round(vals[0], 1),
        )

    def row(self, label: str) -> str:
        return (
            f"  {label:<22} min {self.minimum:>7,.0f}   p50 {self.p50:>7,.0f}   "
            f"p90 {self.p90:>7,.0f}   p95 {self.p95:>7,.0f}   max {self.maximum:>8,.0f}"
        )


@dataclass
class Report:
    dataset: str = ""
    generated_at: str = field(default_factory=now_iso)
    generated_by: str = ""
    max_seq_len: int = 2048

    total: int = 0
    train: int = 0
    eval: int = 0
    quarantined: int = 0

    by_kind: dict[str, int] = field(default_factory=dict)
    by_source: dict[str, int] = field(default_factory=dict)
    by_generator: dict[str, int] = field(default_factory=dict)

    tokens_total: LengthStats = field(default_factory=LengthStats)
    tokens_instruction: LengthStats = field(default_factory=LengthStats)
    tokens_output: LengthStats = field(default_factory=LengthStats)
    scores: LengthStats = field(default_factory=LengthStats)

    over_limit: int = 0
    over_limit_pct: float = 0.0
    longest_examples: list[dict] = field(default_factory=list)

    duplicate_instructions: int = 0
    near_duplicate_instructions: int = 0
    leaked_between_splits: int = 0
    leaked_examples: list[str] = field(default_factory=list)

    degenerate: dict[str, int] = field(default_factory=dict)
    human_authored: int = 0
    estimated_train_tokens: int = 0

    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _tally(items: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _source_label(rec: Record) -> str:
    for tag in rec.tags:
        if tag.startswith("source:"):
            return tag.split(":", 1)[1]
    if rec.source_url.startswith(("human://", "input://")):
        return rec.source_url.split("://")[0]
    return "unknown"


def check_leakage(train: list[Record], evalset: list[Record]) -> tuple[int, list[str]]:
    """Questions appearing in both splits, exactly or near-identically.

    Document-level splitting should prevent this, but two different documents
    can describe the same thing and produce the same question — and that still
    inflates your eval score.
    """
    if not train or not evalset:
        return 0, []

    exact = {normalize_for_hash(r.instruction) for r in train}
    index = _SimhashIndex()
    for rec in train:
        h = simhash(rec.instruction)
        if h:
            index.add(rec.id, h)

    hits: list[str] = []
    for rec in evalset:
        norm = normalize_for_hash(rec.instruction)
        if norm in exact:
            hits.append(rec.instruction)
            continue
        h = simhash(rec.instruction)
        if h and index.query(h, 3):
            hits.append(rec.instruction)

    return len(hits), [truncate(h, 90) for h in hits[:5]]


def find_duplicates(records: list[Record]) -> tuple[int, int]:
    seen: set[str] = set()
    index = _SimhashIndex()
    exact = near = 0
    for rec in records:
        norm = normalize_for_hash(rec.instruction)
        if norm in seen:
            exact += 1
            continue
        seen.add(norm)
        h = simhash(rec.instruction)
        if h:
            if index.query(h, 3):
                near += 1
            else:
                index.add(rec.id, h)
    return exact, near


def find_degenerate(records: list[Record]) -> dict[str, int]:
    """Rows that will teach the model something you did not intend."""
    out = {
        "empty_output": 0,
        "very_short_output": 0,
        "answer_looks_truncated": 0,
        "unbalanced_code_fence": 0,
        "instruction_in_output": 0,
    }
    for rec in records:
        out_text = rec.output.strip()
        if not out_text:
            out["empty_output"] += 1
            continue
        if len(out_text) < 40:
            out["very_short_output"] += 1
        # No terminal punctuation and no closing fence: usually a cut-off answer.
        if not out_text.endswith((".", "!", "?", "`", ")", "]", "}", ":", '"')):
            out["answer_looks_truncated"] += 1
        if out_text.count("```") % 2:
            out["unbalanced_code_fence"] += 1
        if rec.instruction and rec.instruction.lower()[:50] in out_text.lower():
            out["instruction_in_output"] += 1
    return {k: v for k, v in out.items() if v}


def analyze(
    records: list[Record],
    cfg: Config,
    *,
    train: list[Record] | None = None,
    evalset: list[Record] | None = None,
    quarantined: int = 0,
    max_seq_len: int = 2048,
) -> Report:
    from .exporters import split_by_document

    live = [r for r in records if not r.quarantined]
    if train is None or evalset is None:
        train, evalset = split_by_document(
            live, cfg.export.train_split, cfg.export.shuffle_seed
        )

    rep = Report(
        dataset=cfg.name,
        generated_by=f"local model {cfg.llm.model} via {cfg.llm.provider}",
        max_seq_len=max_seq_len,
        total=len(live),
        train=len(train),
        eval=len(evalset),
        quarantined=quarantined,
    )
    if not live:
        rep.warnings.append("The dataset is empty — run `datagen build` first.")
        return rep

    rep.by_kind = _tally(r.kind for r in live)
    rep.by_source = _tally(_source_label(r) for r in live)
    rep.by_generator = _tally((r.generator or "unknown").split(":")[0] for r in live)
    rep.human_authored = sum(1 for r in live if r.generator.startswith("human"))

    instruction_tokens = [estimate_tokens(r.instruction) for r in live]
    output_tokens = [estimate_tokens(r.output) for r in live]
    total_tokens = [i + o for i, o in zip(instruction_tokens, output_tokens)]

    rep.tokens_instruction = LengthStats.of(instruction_tokens)
    rep.tokens_output = LengthStats.of(output_tokens)
    rep.tokens_total = LengthStats.of(total_tokens)
    rep.scores = LengthStats.of([r.score for r in live if r.score is not None])
    rep.estimated_train_tokens = sum(
        estimate_tokens(r.instruction) + estimate_tokens(r.output) for r in train
    )

    over = [(t, r) for t, r in zip(total_tokens, live) if t > max_seq_len]
    rep.over_limit = len(over)
    rep.over_limit_pct = round(100 * len(over) / len(live), 2)
    rep.longest_examples = [
        {"tokens": t, "kind": r.kind, "instruction": truncate(r.instruction, 80),
         "source": r.source_url}
        for t, r in sorted(over, key=lambda x: -x[0])[:5]
    ]

    rep.duplicate_instructions, rep.near_duplicate_instructions = find_duplicates(live)
    rep.leaked_between_splits, rep.leaked_examples = check_leakage(train, evalset)
    rep.degenerate = find_degenerate(live)

    _add_warnings(rep, cfg)
    return rep


def _add_warnings(rep: Report, cfg: Config) -> None:
    w, n = rep.warnings, rep.notes

    if rep.total < MIN_VIABLE_RECORDS:
        w.append(
            f"Only {rep.total} records. A supervised fine-tune realistically wants "
            f"{MIN_USEFUL_RECORDS}+; below ~{MIN_VIABLE_RECORDS} you are more likely "
            "to damage the base model than teach it. Add sources, or use this as a "
            "RAG corpus (rag_chunks.jsonl) instead of a fine-tuning set."
        )
    elif rep.total < MIN_USEFUL_RECORDS:
        w.append(
            f"{rep.total} records is on the small side for fine-tuning "
            f"(~{MIN_USEFUL_RECORDS}+ is a more comfortable floor). Consider a LoRA "
            "with a low rank and few epochs rather than a full fine-tune."
        )

    if rep.over_limit:
        w.append(
            f"{rep.over_limit} rows ({rep.over_limit_pct}%) exceed max_seq_len="
            f"{rep.max_seq_len} and will be TRUNCATED during training, teaching the "
            f"model to stop mid-answer. Either raise max_seq_len to "
            f"{int(rep.tokens_total.p99) + 64} (covers p99) or lower "
            "`quality.max_answer_chars` and regenerate."
        )

    if rep.leaked_between_splits:
        w.append(
            f"{rep.leaked_between_splits} questions appear in BOTH train and eval. "
            "Your eval loss will look better than reality. This usually means two "
            "sources cover the same material — dedupe harder "
            "(raise `dedupe.simhash_bits`) or accept a smaller eval set."
        )

    if rep.duplicate_instructions or rep.near_duplicate_instructions:
        n.append(
            f"{rep.duplicate_instructions} exact and "
            f"{rep.near_duplicate_instructions} near-duplicate questions remain. "
            "Some repetition is fine; a lot means over-weighted rows."
        )

    if rep.eval == 0:
        w.append(
            "The eval split is empty — you cannot measure whether training helped. "
            "This happens when everything comes from one source document."
        )
    elif rep.eval < 20 and rep.total > 100:
        n.append(f"Eval split is only {rep.eval} rows; treat its metrics as indicative.")

    if rep.by_kind:
        top_kind, top_n = next(iter(rep.by_kind.items()))
        if top_n / rep.total > 0.8 and len(rep.by_kind) > 1:
            n.append(
                f"{round(100 * top_n / rep.total)}% of rows are '{top_kind}'. The model "
                "will learn that format strongly and others weakly."
            )

    if rep.by_source:
        top_source, top_n = next(iter(rep.by_source.items()))
        if top_n / rep.total > 0.7 and len(rep.by_source) > 1:
            n.append(
                f"{round(100 * top_n / rep.total)}% of rows come from '{top_source}'. "
                "Broaden sources to avoid teaching one document's voice."
            )

    if rep.scores.count and rep.scores.mean < 0.75:
        n.append(
            f"Mean quality score is {rep.scores.mean}. Inspect what is scraping through: "
            "`datagen inspect -n 10`."
        )

    truncated = rep.degenerate.get("answer_looks_truncated", 0)
    if truncated and truncated / rep.total > 0.15:
        w.append(
            f"{truncated} answers ({round(100 * truncated / rep.total)}%) end without "
            "terminal punctuation — they were probably cut off by the model's output "
            "limit. Raise `llm.num_ctx` or lower `generation.pairs_per_chunk`."
        )

    if rep.degenerate.get("empty_output"):
        w.append(f"{rep.degenerate['empty_output']} rows have an EMPTY output. Drop them.")

    if rep.human_authored:
        n.append(
            f"{rep.human_authored} rows are human-authored (`datagen learn`) and were "
            "not machine-judged. These are usually your highest-value rows."
        )

    if not w:
        n.append("No blocking problems found.")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def report_text(rep: Report) -> str:
    L: list[str] = []
    L.append(f"\n  Dataset analysis — {rep.dataset}")
    L.append("  " + "─" * 76)
    L.append(f"  {rep.total:,} records   train {rep.train:,} / eval {rep.eval:,}"
             f"   quarantined {rep.quarantined:,}")
    L.append(f"  generated by {rep.generated_by}")
    L.append(f"  ~{rep.estimated_train_tokens:,} training tokens (estimated)")

    L.append("\n  TOKEN LENGTHS (estimated — verify with your trainer's tokenizer)")
    L.append(rep.tokens_instruction.row("instruction"))
    L.append(rep.tokens_output.row("output"))
    L.append(rep.tokens_total.row("instruction+output"))
    verdict = "OK" if not rep.over_limit else f"{rep.over_limit} WILL TRUNCATE"
    L.append(f"\n  at max_seq_len={rep.max_seq_len}: {verdict} ({rep.over_limit_pct}%)")
    for ex in rep.longest_examples:
        L.append(f"      {ex['tokens']:>6,} tok  [{ex['kind']}]  {ex['instruction']}")

    L.append("\n  COMPOSITION")
    for label, table in (("by kind", rep.by_kind), ("by source", rep.by_source),
                         ("by generator", rep.by_generator)):
        if table:
            parts = ", ".join(f"{k} {v:,} ({round(100 * v / rep.total)}%)"
                              for k, v in list(table.items())[:6])
            L.append(f"    {label:<14} {parts}")

    if rep.scores.count:
        L.append(f"    quality score  mean {rep.scores.mean}  p50 {rep.scores.p50}  "
                 f"min {rep.scores.minimum}")

    L.append("\n  INTEGRITY")
    L.append(f"    duplicate questions      {rep.duplicate_instructions} exact, "
             f"{rep.near_duplicate_instructions} near")
    L.append(f"    train/eval leakage       {rep.leaked_between_splits}")
    for ex in rep.leaked_examples:
        L.append(f"        {ex}")
    if rep.degenerate:
        L.append("    degenerate rows          " +
                 ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in rep.degenerate.items()))

    if rep.warnings:
        L.append("\n  WARNINGS")
        for item in rep.warnings:
            L.append("    ! " + _wrap(item, 74, "      "))
    if rep.notes:
        L.append("\n  NOTES")
        for item in rep.notes:
            L.append("    - " + _wrap(item, 74, "      "))

    L.append("")
    return "\n".join(L)


def _wrap(text: str, width: int, indent: str) -> str:
    import textwrap

    return ("\n" + indent).join(textwrap.wrap(text, width))


def dataset_card(rep: Report, cfg: Config) -> str:
    """A dataset card to keep next to a trained checkpoint.

    The point is reproducibility: six months from now this should answer "where
    did this data come from, what was it good for, and what was wrong with it".
    """
    def table(d: dict[str, int]) -> str:
        if not d:
            return "_none_"
        return "\n".join(
            f"| {k} | {v:,} | {round(100 * v / max(1, rep.total), 1)}% |"
            for k, v in d.items()
        )

    warn_block = (
        "\n".join(f"- {w}" for w in rep.warnings)
        if rep.warnings
        else "- No blocking problems were detected."
    )

    return f"""# Dataset Card — {rep.dataset}

{cfg.description or ""}

- **Created:** {rep.generated_at}
- **Records:** {rep.total:,} ({rep.train:,} train / {rep.eval:,} eval)
- **Estimated training tokens:** ~{rep.estimated_train_tokens:,}
- **Generated by:** {rep.generated_by} — a **local** model. No third-party API
  received any of the source material.
- **Quarantined (excluded):** {rep.quarantined:,}

## How it was built

Source documents were parsed, split into heading-aware chunks, deduplicated in
three passes (exact hash, SimHash near-duplicate, embedding cosine), and turned
into instruction/response pairs by the local model above. Each row was then
scored by heuristics, a grounding check against its own source chunk, and an
LLM judge; rows below `quality.min_score = {cfg.quality.min_score}` were
quarantined rather than deleted.

Every row carries the URL and title of the source it came from.

## Composition

### By kind
| kind | rows | share |
|---|---:|---:|
{table(rep.by_kind)}

### By source
| source | rows | share |
|---|---:|---:|
{table(rep.by_source)}

## Length statistics

Token counts are **estimates** (character- and word-based heuristics, no
tokenizer dependency). Verify against your trainer's tokenizer before fixing
`max_seq_len`.

| field | p50 | p90 | p95 | max |
|---|---:|---:|---:|---:|
| instruction | {rep.tokens_instruction.p50:,.0f} | {rep.tokens_instruction.p90:,.0f} | {rep.tokens_instruction.p95:,.0f} | {rep.tokens_instruction.maximum:,.0f} |
| output | {rep.tokens_output.p50:,.0f} | {rep.tokens_output.p90:,.0f} | {rep.tokens_output.p95:,.0f} | {rep.tokens_output.maximum:,.0f} |
| combined | {rep.tokens_total.p50:,.0f} | {rep.tokens_total.p90:,.0f} | {rep.tokens_total.p95:,.0f} | {rep.tokens_total.maximum:,.0f} |

At `max_seq_len = {rep.max_seq_len}`, **{rep.over_limit} rows
({rep.over_limit_pct}%)** would be truncated. A limit of
**{int(rep.tokens_total.p99) + 64}** covers the 99th percentile.

## Splits

Split **by source document**, not by row: all rows derived from one document
land on the same side. Rows from a single chunk are near-duplicates of each
other, so a row-level split would leak train content into eval.

- Detected train/eval question overlap: **{rep.leaked_between_splits}**

## Intended use

Supervised fine-tuning (LoRA or full) and retrieval-augmented generation. The
`rag_chunks.jsonl` export is the retrieval corpus; the alpaca/sharegpt/chatml
exports are the SFT sets.

## Limitations and risks

{warn_block}

Additional caveats that apply to any dataset built this way:

- **Synthetic.** Rows were written by a language model from source text. The
  grounding check reduces invention but does not eliminate it — spot-check
  before trusting a fine-tune on anything safety-relevant.
- **Source licensing is not verified.** Crawled web pages may not permit use as
  training data. Check the licence of every source in the table above before
  training a model you intend to distribute.
- **Inherits source bias.** If the sources are wrong, outdated, or one-sided,
  the dataset is too, with the errors now stated confidently.
- **Not human-reviewed** except for the {rep.human_authored} rows authored
  directly via `datagen learn`.

## Reproducing

```bash
python -m datagen build --full
python -m datagen analyze
```

Configuration lives in `config.toml`. Quality threshold
`{cfg.quality.min_score}`, chunk size `{cfg.chunking.max_chars}`, generation
kinds `{", ".join(cfg.generation.kinds)}`.
"""


def load_records(path: Path) -> list[Record]:
    """Read records back from a JSONL export."""
    if not path.exists():
        return []
    out: list[Record] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(Record(**{k: v for k, v in data.items() if k in Record.__annotations__}))
    return out
