"""Quality gate.

Two layers, cheap first:

1. Heuristics — length bounds, degenerate questions, answers that are just the
   question restated, refusal/meta phrasing ("the text does not say"), copy of
   navigation chrome, unbalanced code fences. These catch most of what a 7B
   model gets wrong and cost nothing.

2. Grounding + judge — token-overlap grounding against the source chunk, then
   (optionally) the local model scoring the pair on faithfulness, usefulness and
   self-containment.

Failing records are QUARANTINED, not deleted: they are written to
`data/quarantine.jsonl` with a reason so you can inspect what the gate rejected
and tune `quality.min_score` rather than flying blind.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import QualityConfig
from .llm import LocalLLM
from .models import Record
from .util import get_logger, tokenize, truncate

log = get_logger("quality")

# Phrases that mean the model answered about the prompt instead of the topic.
_META_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"\bthe (?:given |provided )?(?:source|text|document|passage|context)\b",
        r"\baccording to the (?:above|text|source|document)\b",
        r"\bas (?:mentioned|stated|described) (?:above|in the text)\b",
        r"\bi (?:do not|don't) have (?:enough|the) (?:information|context)\b",
        r"\b(?:is |are )?not (?:specified|mentioned|provided|stated) in the\b",
        r"^\s*(?:n/?a|none|unknown|no answer)\s*$",
    )
]
_NAV_NOISE = re.compile(
    r"(cookie|privacy policy|all rights reserved|skip to (?:main )?content|"
    r"sign in|subscribe|newsletter|©\s*\d{4})",
    re.I,
)

_JUDGE_PROMPT = """Score this training example against its SOURCE.

Score each 0.0-1.0:
- faithful: is every claim in the ANSWER supported by the SOURCE? (the most
  important criterion — a confident wrong answer scores 0)
- useful: would this teach an engineer something actionable?
- standalone: does the QUESTION make sense without seeing the SOURCE?

Return JSON only:
{{"faithful": 0.0, "useful": 0.0, "standalone": 0.0, "reason": "<12 words"}}

QUESTION: {question}
ANSWER: {answer}
SOURCE:
\"\"\"
{source}
\"\"\""""


@dataclass
class Verdict:
    ok: bool
    score: float
    reason: str


def heuristic_check(rec: Record, cfg: QualityConfig) -> Verdict:
    q, a = rec.instruction.strip(), rec.output.strip()

    if len(q) < cfg.min_question_chars:
        return Verdict(False, 0.0, "question too short")
    if len(a) < cfg.min_answer_chars:
        return Verdict(False, 0.0, "answer too short")
    if len(a) > cfg.max_answer_chars:
        return Verdict(False, 0.2, f"answer too long ({len(a)} chars)")
    if q.lower() == a.lower() or a.lower().startswith(q.lower()[:60]) and len(a) < len(q) * 1.5:
        return Verdict(False, 0.0, "answer restates the question")

    for pat in _META_PATTERNS:
        if pat.search(q):
            return Verdict(False, 0.1, "question refers to the source text")
        if pat.search(a[:200]):
            return Verdict(False, 0.1, "answer is meta or a non-answer")

    if _NAV_NOISE.search(a) and len(a) < 400:
        return Verdict(False, 0.1, "answer is page chrome")
    if a.count("```") % 2:
        return Verdict(False, 0.3, "unbalanced code fence")
    if q.count("?") > 3:
        return Verdict(False, 0.3, "question is a multi-question blob")

    # Answers that are one long unbroken token stream are usually extraction junk.
    words = a.split()
    if words and sum(len(w) for w in words) / len(words) > 18:
        return Verdict(False, 0.2, "answer looks like garbled extraction")

    return Verdict(True, 0.75, "heuristics ok")


# `<pod-name>`, `{namespace}` — placeholders the answer invents by design. They
# are not claims about the source and must not count against grounding.
_PLACEHOLDER = re.compile(r"[<{\[][a-z0-9_.\- ]{2,40}[>}\]]", re.I)


def _norm_tokens(text: str) -> set[str]:
    """Tokens with sentence punctuation stripped.

    The shared tokenizer keeps '.' inside tokens so that `v1.2` and `foo.yaml`
    survive — which also means "ready." never matches "ready" in the source.
    """
    return {t for t in (tok.strip(".,;:!?()") for tok in tokenize(text)) if t}


def _is_identifier(token: str) -> bool:
    """Commands, flags, paths, versions, error codes — the tokens a model
    hallucinates when it goes off-source.

    Tested on an already-stripped token: otherwise every sentence-final word
    ("ready.", "failing.") looks like a dotted identifier and a perfectly
    grounded answer gets marked as invented.
    """
    return any(c.isdigit() for c in token) or any(c in token for c in "/-_.")


def grounding_score(rec: Record) -> float:
    """How much of the answer is actually supported by its source chunk.

    Plain token overlap is the obvious metric and it is wrong: it scores a
    verbatim copy 1.0 and a well-written explanation 0.2, because words like
    "cause", "diagnose" and "verify" are what an answer adds and a source rarely
    repeats. Training on the copies and discarding the explanations is the exact
    opposite of what this dataset is for.

    What actually indicates an off-source answer is an *identifier* that is not
    in the source — a command, a flag, a path, an error code that the model
    invented. So identifiers are scored strictly and prose leniently.
    """
    if not rec.context:
        return 1.0
    answer_tokens = _norm_tokens(_PLACEHOLDER.sub(" ", rec.output))
    if not answer_tokens:
        return 0.0
    source_tokens = _norm_tokens(rec.context)
    overlap = len(answer_tokens & source_tokens) / len(answer_tokens)

    hard = {t for t in answer_tokens if _is_identifier(t)}
    if not hard:
        # No technical claims to verify: prose overlap is all we have.
        return overlap
    hard_overlap = len(hard & source_tokens) / len(hard)
    return max(overlap, hard_overlap * 0.75 + overlap * 0.25)


def judge(rec: Record, llm: LocalLLM) -> Verdict | None:
    """Local-model judgement. Returns None when the model is unavailable."""
    data = llm.complete_json(
        _JUDGE_PROMPT.format(
            question=truncate(rec.instruction, 600),
            answer=truncate(rec.output, 2500),
            source=truncate(rec.context, 5000),
        ),
        system="You are a strict dataset reviewer. Reply with JSON only.",
        temperature=0.0,
    )
    if not isinstance(data, dict):
        return None

    def num(key: str) -> float:
        try:
            return max(0.0, min(1.0, float(data.get(key, 0))))
        except (TypeError, ValueError):
            return 0.0

    faithful, useful, standalone = num("faithful"), num("useful"), num("standalone")
    score = faithful * 0.55 + useful * 0.25 + standalone * 0.20
    reason = str(data.get("reason", ""))[:120]
    return Verdict(True, round(score, 3), reason or "judged")


def evaluate(records: list[Record], llm: LocalLLM, cfg: QualityConfig) -> tuple[list[Record], list[Record]]:
    """Score every record. Returns (accepted, quarantined)."""
    if not cfg.enabled:
        return records, []

    accepted: list[Record] = []
    quarantined: list[Record] = []
    use_judge = llm.available()
    if not use_judge:
        log.info("quality: local model unavailable — heuristics + grounding only")

    for rec in records:
        verdict = heuristic_check(rec, cfg)

        if verdict.ok and cfg.require_grounded:
            g = grounding_score(rec)
            if g < 0.35:
                verdict = Verdict(False, round(g, 3), f"ungrounded (overlap {g:.2f})")
            else:
                verdict = Verdict(True, min(1.0, 0.5 + g * 0.5), "grounded")

        # Only pay for the judge on records that already passed the cheap checks.
        if verdict.ok and use_judge:
            judged = judge(rec, llm)
            if judged is not None:
                verdict = Verdict(judged.score >= cfg.min_score, judged.score, judged.reason)
            elif not llm.available():
                use_judge = False

        rec.score = verdict.score
        rec.score_reason = verdict.reason
        if verdict.ok and verdict.score >= cfg.min_score:
            accepted.append(rec)
        else:
            rec.quarantined = True
            quarantined.append(rec)

    log.info(
        "quality: %d accepted, %d quarantined (threshold %.2f)",
        len(accepted), len(quarantined), cfg.min_score,
    )
    if quarantined:
        reasons: dict[str, int] = {}
        for r in quarantined:
            reasons[r.score_reason] = reasons.get(r.score_reason, 0) + 1
        top = sorted(reasons.items(), key=lambda kv: -kv[1])[:5]
        log.info("  top rejection reasons: %s", ", ".join(f"{k} ×{v}" for k, v in top))

    return accepted, quarantined
