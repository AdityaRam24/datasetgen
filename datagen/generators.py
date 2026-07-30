"""Record generation: chunk -> training rows, using the LOCAL model.

Three kinds, each with its own prompt:
  qa               general questions a user would actually ask
  instruction      imperative tasks ("Configure X so that Y")
  troubleshooting  symptom -> diagnosis -> fix, the highest-value kind for an
                   ops assistant; strongly preferred for runbooks and error docs

Every kind has an EXTRACTIVE FALLBACK that runs with no model at all (heading
-> question, body -> answer). That is what keeps the pipeline useful when
Ollama isn't running, and it is why `datagen build` never hard-fails.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from .config import Config
from .llm import LocalLLM
from .models import Chunk, Record, RunStats
from .util import clean_text, get_logger, truncate

log = get_logger("generators")

SYSTEM = (
    "You are a meticulous technical dataset author. You write training examples "
    "for an assistant that supports HPE Private Cloud AI and general platform "
    "operations. You never invent facts: every answer must be fully supported by "
    "the SOURCE text you are given. If the source does not answer something, you "
    "do not write that example. You always reply with valid JSON and nothing else."
)

_QA_PROMPT = """Read the SOURCE below and write {n} question/answer pairs.

Rules:
- Questions must be ones a real engineer would type, self-contained (never
  "according to the text above" or "what does this document say").
- Answers must be complete, technical, and drawn ONLY from the SOURCE.
- Preserve exact command names, flags, paths, ports, and error strings.
- If the SOURCE is boilerplate (navigation, a copyright notice, a table of
  contents) return {{"pairs": []}}.

Return JSON exactly like:
{{"pairs": [{{"question": "...", "answer": "..."}}]}}

TITLE: {title}
SOURCE:
\"\"\"
{text}
\"\"\""""

_INSTRUCTION_PROMPT = """Read the SOURCE and write {n} instruction/response pairs
suitable for supervised fine-tuning.

Rules:
- The instruction is an imperative task ("Configure...", "Explain how to...",
  "List the prerequisites for..."), not a yes/no question.
- The response is a complete, actionable answer grounded ONLY in the SOURCE.
- Include concrete commands and config snippets when the SOURCE has them.
- Return {{"pairs": []}} if the SOURCE has no actionable content.

Return JSON exactly like:
{{"pairs": [{{"instruction": "...", "response": "..."}}]}}

TITLE: {title}
SOURCE:
\"\"\"
{text}
\"\"\""""

_TROUBLESHOOT_PROMPT = """Read the SOURCE and write up to {n} TROUBLESHOOTING
examples: a realistic failure a user reports, and the diagnosis + fix.

Rules:
- "symptom" should read like a real report — an error message, a stuck state, a
  failed command. Quote real error strings from the SOURCE when present.
- "answer" must give: the likely cause, the diagnostic commands to confirm it,
  and the ordered remediation steps. Ground everything in the SOURCE.
- Preserve the step ORDER if the SOURCE is a procedure.
- If the SOURCE describes no failure mode, error, limitation or recovery
  procedure, return {{"cases": []}}. Do NOT invent failures.

Return JSON exactly like:
{{"cases": [{{"symptom": "...", "answer": "..."}}]}}

TITLE: {title}
SOURCE:
\"\"\"
{text}
\"\"\""""


# ---------------------------------------------------------------------------
# LLM-backed generators
# ---------------------------------------------------------------------------


def _gen_qa(chunk: Chunk, llm: LocalLLM, n: int) -> list[Record]:
    data = llm.complete_json(
        _QA_PROMPT.format(n=n, title=chunk.title, text=truncate(chunk.text, 6000)),
        system=SYSTEM,
    )
    return _pairs_to_records(data, chunk, "qa", ("question", "answer"), llm.cfg.model, "pairs")


def _gen_instruction(chunk: Chunk, llm: LocalLLM, n: int) -> list[Record]:
    data = llm.complete_json(
        _INSTRUCTION_PROMPT.format(n=n, title=chunk.title, text=truncate(chunk.text, 6000)),
        system=SYSTEM,
    )
    return _pairs_to_records(
        data, chunk, "instruction", ("instruction", "response"), llm.cfg.model, "pairs"
    )


def _gen_troubleshooting(chunk: Chunk, llm: LocalLLM, n: int) -> list[Record]:
    data = llm.complete_json(
        _TROUBLESHOOT_PROMPT.format(
            n=max(1, n - 1), title=chunk.title, text=truncate(chunk.text, 6000)
        ),
        system=SYSTEM,
    )
    return _pairs_to_records(
        data, chunk, "troubleshooting", ("symptom", "answer"), llm.cfg.model, "cases"
    )


def _pairs_to_records(
    data: Any,
    chunk: Chunk,
    kind: str,
    keys: tuple[str, str],
    model: str,
    list_key: str,
) -> list[Record] | None:
    """Tolerant parsing — small models return the list under all sorts of keys."""
    if data is None:
        return None  # signals "LLM failed", distinct from "no examples found"

    items: list = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for candidate in (list_key, "pairs", "cases", "examples", "items", "data", "results"):
            if isinstance(data.get(candidate), list):
                items = data[candidate]
                break
        else:
            # A single object rather than a list.
            if any(k in data for k in keys):
                items = [data]

    out: list[Record] = []
    qk, ak = keys
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get(qk) or item.get("question") or item.get("instruction") or "").strip()
        answer = str(item.get(ak) or item.get("answer") or item.get("response") or item.get("output") or "").strip()
        if not question or not answer:
            continue
        out.append(
            Record.make(
                kind, clean_text(question), clean_text(answer), chunk, generator=f"llm:{model}"
            )
        )
    return out


# ---------------------------------------------------------------------------
# Extractive fallbacks — no model required
# ---------------------------------------------------------------------------

_STEP_LINE = re.compile(r"^\s*(\d+)[.)]\s+(.+)$")
_ERROR_LINE = re.compile(
    r"(error|failed|failure|cannot|denied|timeout|timed out|refused|unavailable|"
    r"crashloopbackoff|imagepullbackoff|oomkilled|not found|exception)",
    re.I,
)


def _extractive(chunk: Chunk, kinds: list[str]) -> list[Record]:
    """Deterministic records built from structure alone.

    Lower quality than a model, but grounded by construction: the answer IS the
    source text, so it can never hallucinate.
    """
    out: list[Record] = []
    heading = chunk.heading.split(" › ")[-1].strip() if chunk.heading else ""
    body = chunk.text
    if heading and body.startswith(chunk.heading):
        body = body[len(chunk.heading) :].strip()
    body = clean_text(body)
    if len(body) < 80:
        return out

    topic = heading or chunk.title

    if "qa" in kinds and topic:
        out.append(
            Record.make(
                "qa",
                f"What does the {chunk.title} documentation say about {topic}?",
                body,
                chunk,
                generator="extractive:heading",
            )
        )

    steps = [m.group(2).strip() for m in (_STEP_LINE.match(l) for l in body.split("\n")) if m]
    if "instruction" in kinds and len(steps) >= 2:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))
        out.append(
            Record.make(
                "instruction",
                f"Walk me through the procedure for {topic}.",
                numbered,
                chunk,
                generator="extractive:steps",
            )
        )

    if "troubleshooting" in kinds:
        error_lines = [l.strip() for l in body.split("\n") if _ERROR_LINE.search(l) and len(l) > 25]
        if error_lines:
            symptom = truncate(error_lines[0], 220)
            out.append(
                Record.make(
                    "troubleshooting",
                    f"I am seeing: {symptom}\nWhat does this mean and how do I fix it?",
                    body,
                    chunk,
                    generator="extractive:error-line",
                )
            )

    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

GENERATORS: dict[str, Callable[[Chunk, LocalLLM, int], list[Record] | None]] = {
    "qa": _gen_qa,
    "instruction": _gen_instruction,
    "troubleshooting": _gen_troubleshooting,
}


def _kinds_for(chunk: Chunk, configured: list[str]) -> list[str]:
    """Bias the kind mix by material: runbooks and error pages deserve
    troubleshooting examples; reference pages mostly want Q&A."""
    kinds = [k for k in configured if k in GENERATORS]
    is_procedural = chunk.kind == "runbook" or "runbook" in chunk.tags
    has_errors = bool(_ERROR_LINE.search(chunk.text))

    if is_procedural or has_errors:
        if "troubleshooting" in kinds:
            kinds = ["troubleshooting"] + [k for k in kinds if k != "troubleshooting"]
    else:
        kinds = [k for k in kinds if k != "troubleshooting"] or kinds
    return kinds


def generate_for_chunk(
    chunk: Chunk, llm: LocalLLM, cfg: Config, stats: RunStats | None = None
) -> list[Record]:
    kinds = _kinds_for(chunk, cfg.generation.kinds)
    per_kind = max(1, cfg.generation.pairs_per_chunk // max(1, len(kinds)))

    records: list[Record] = []
    llm_worked = False

    for kind in kinds:
        result = GENERATORS[kind](chunk, llm, per_kind) if llm.available() else None
        if result is None:
            continue                       # LLM unavailable or call failed
        llm_worked = True
        records.extend(result)

    if not llm_worked:
        if not cfg.llm.allow_extractive_fallback:
            raise RuntimeError(
                f"local LLM unavailable at {cfg.llm.base_url} and "
                "llm.allow_extractive_fallback is false"
            )
        records = _extractive(chunk, kinds)
        if stats:
            stats.extractive_fallbacks += 1

    if cfg.generation.include_citations:
        for r in records:
            r.tags = list(dict.fromkeys(r.tags + [f"source:{chunk.source}"]))

    return records


def generate_all(
    chunks: list[Chunk], llm: LocalLLM, cfg: Config, stats: RunStats
) -> list[Record]:
    """Generate across all chunks, in parallel where the local model allows.

    Note the worker count: a local model is usually one GPU serving one process,
    so more than ~4 concurrent requests makes things slower, not faster.
    """
    if not chunks:
        return []

    workers = max(1, min(cfg.generation.max_workers, 8))
    log.info(
        "generating from %d chunks (%s, %d workers)",
        len(chunks),
        f"local model {cfg.llm.model}" if llm.available() else "extractive fallback",
        workers,
    )

    out: list[Record] = []
    done = 0

    def work(chunk: Chunk) -> list[Record]:
        try:
            return generate_for_chunk(chunk, llm, cfg, stats)
        except RuntimeError:
            raise
        except Exception as e:
            log.warning("generation failed for chunk %s: %s", chunk.id[:8], e)
            stats.errors.append(f"generate {chunk.id[:8]}: {e}")
            return []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for records in pool.map(work, chunks):
            out.extend(records)
            done += 1
            if done % 25 == 0 or done == len(chunks):
                log.info("  %d/%d chunks -> %d records", done, len(chunks), len(out))

    stats.llm_calls = llm.stats.calls
    stats.llm_failures = llm.stats.failures
    return out
