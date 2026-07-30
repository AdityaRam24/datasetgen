"""Runbook connector.

Runbooks are procedures, and procedures lose their meaning when they are
chopped up naively — "run the drain command" is useless without knowing which
failure it belongs to and what must be true first.

So this connector parses runbook structure explicitly:
  * symptom / trigger / precondition sections
  * ordered steps (1. 2. 3., `- [ ]`, `Step N:`)
  * commands in fenced blocks or `$`-prefixed lines
  * verification and rollback sections

and re-emits a normalised document where the ordering is explicit. Generators
then produce troubleshooting records that keep the sequence intact.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..models import Document
from ..util import clean_text, get_logger
from .files import iter_files, read_file

log = get_logger("runbooks")

DEFAULT_GLOBS = ["**/*.md", "**/*.txt", "**/*.pdf", "**/*.docx"]

SECTION_ALIASES = {
    "symptom": ("symptom", "symptoms", "problem", "issue", "error", "alert", "when to use"),
    "cause": ("cause", "causes", "root cause", "why", "background", "diagnosis"),
    "precondition": ("precondition", "preconditions", "prerequisite", "prerequisites",
                     "requirements", "before you begin", "assumptions"),
    "steps": ("steps", "procedure", "resolution", "remediation", "fix", "how to",
              "instructions", "actions", "workaround", "mitigation"),
    "verify": ("verify", "verification", "validation", "confirm", "post-checks",
               "success criteria", "testing"),
    "rollback": ("rollback", "roll back", "revert", "undo", "backout", "recovery"),
    "escalate": ("escalate", "escalation", "contact", "owner", "on-call", "support"),
}

_HEADING = re.compile(r"^\s{0,3}(?:#{1,6}\s*|\*\*)?([A-Za-z][A-Za-z /\-]{2,40})\**\s*:?\s*$")
_STEP = re.compile(r"^\s*(?:(\d+)[.)]\s+|[-*]\s+\[.\]\s*|[-*]\s+|Step\s+(\d+)\s*[:.\-]\s*)(.+)$", re.I)
_FENCE = re.compile(r"```[\w]*\n(.*?)```", re.S)
_SHELL = re.compile(r"^\s*[\$#>]\s*(\S.*)$")


def classify_heading(text: str) -> str | None:
    low = text.strip().lower().rstrip(":")
    for canonical, aliases in SECTION_ALIASES.items():
        if any(low == a or low.startswith(a) for a in aliases):
            return canonical
    return None


def parse_runbook(text: str, title: str) -> dict[str, Any]:
    """Extract structure. Returns a dict of section -> content plus `steps` and
    `commands` lists. Anything unrecognised lands in `body`."""
    sections: dict[str, list[str]] = {}
    current = "body"
    commands: list[str] = []

    for block in _FENCE.findall(text):
        for line in block.strip().split("\n"):
            if line.strip() and not line.strip().startswith("#"):
                commands.append(line.strip().lstrip("$ ").strip())

    for line in text.split("\n"):
        m = _HEADING.match(line)
        if m:
            canonical = classify_heading(m.group(1))
            if canonical:
                current = canonical
                continue
        sections.setdefault(current, []).append(line)
        sh = _SHELL.match(line)
        if sh:
            commands.append(sh.group(1).strip())

    steps: list[str] = []
    step_source = sections.get("steps") or sections.get("body") or []
    for line in step_source:
        m = _STEP.match(line)
        if m and m.group(3).strip():
            steps.append(m.group(3).strip())

    out: dict[str, Any] = {
        k: clean_text("\n".join(v)) for k, v in sections.items() if any(x.strip() for x in v)
    }
    out["steps"] = steps
    out["commands"] = list(dict.fromkeys(c for c in commands if len(c) > 2))[:60]
    out["title"] = title
    return out


def normalize(parsed: dict[str, Any], title: str) -> str:
    """Render the parsed runbook back into a canonical, LLM-friendly layout."""
    lines = [f"# Runbook: {title}", ""]

    for key, label in (
        ("symptom", "Symptom / when this applies"),
        ("cause", "Likely cause"),
        ("precondition", "Preconditions"),
    ):
        if parsed.get(key):
            lines += [f"## {label}", parsed[key], ""]

    steps = parsed.get("steps") or []
    if steps:
        lines.append("## Resolution steps (in order)")
        lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        lines.append("")

    if parsed.get("commands"):
        lines.append("## Commands referenced")
        lines += [f"- `{c}`" for c in parsed["commands"]]
        lines.append("")

    for key, label in (
        ("verify", "Verification"),
        ("rollback", "Rollback"),
        ("escalate", "Escalation"),
    ):
        if parsed.get(key):
            lines += [f"## {label}", parsed[key], ""]

    leftover = parsed.get("body", "")
    if leftover and not steps:
        lines += ["## Notes", leftover, ""]

    return clean_text("\n".join(lines))


def fetch(cfg: Any, block: dict, state: Any = None) -> list[Document]:
    name = block.get("name", "runbooks")
    root = cfg.resolve(block.get("path", "corpus/runbooks"))
    if not root.exists():
        log.warning("[%s] runbook path does not exist: %s", name, root)
        return []

    tags = list(block.get("tags", [])) or ["runbook"]
    docs: list[Document] = []

    for path in iter_files(root, block.get("globs", DEFAULT_GLOBS)):
        raw = read_file(path, name, tags)
        if not raw:
            continue
        parsed = parse_runbook(raw.text, raw.title)
        doc = Document.make(
            title=raw.title,
            url=f"runbook://{path.relative_to(root).as_posix()}",
            text=normalize(parsed, raw.title),
            kind="runbook",
            source=name,
            tags=tags,
            meta={
                "path": str(path),
                "steps": len(parsed.get("steps", [])),
                "commands": parsed.get("commands", [])[:20],
                "has_symptom": bool(parsed.get("symptom")),
                "has_rollback": bool(parsed.get("rollback")),
            },
        )
        docs.append(doc)

    log.info("[%s] parsed %d runbooks", name, len(docs))
    return docs
