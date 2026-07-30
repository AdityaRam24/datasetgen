"""Learn from input you provide.

This is the "teach it something" path. Everything else in the project goes out
and finds material; this takes material you hand it and folds it into the same
dataset and knowledge base.

Four input shapes, in increasing order of how much the model is trusted:

  text / file / stdin   raw material -> chunked, generated, quality-gated like
                        any other source
  url                   fetched, then treated as above
  question + answer     a pair YOU wrote. Not generated, not judged, not
                        paraphrased — stored verbatim and trusted, because a
                        human correction is the highest-value row in the set
  solved case           a problem you hit and how you fixed it, stored as a
                        troubleshooting record

Everything lands with a `input://` or `human://` URL so you can always tell
which rows came from a person rather than a crawl.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

from .config import Config
from .models import Chunk, Document, Record
from .util import clean_text, get_logger, now_iso, short_id, truncate

log = get_logger("learn")


def _slug(text: str, limit: int = 48) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:limit] or "untitled"


def read_stdin() -> str:
    """Read piped input. Returns '' when stdin is an interactive terminal."""
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    try:
        return sys.stdin.read()
    except (OSError, UnicodeDecodeError) as e:
        log.warning("could not read stdin: %s", e)
        return ""


def read_interactive() -> str:
    """Prompt for a multi-line paste."""
    end = "Ctrl-Z then Enter" if sys.platform == "win32" else "Ctrl-D"
    print(f"\n  Paste or type your input. Finish with {end} on a blank line.\n")
    try:
        return sys.stdin.read()
    except KeyboardInterrupt:
        return ""


def document_from_text(
    text: str,
    *,
    title: str = "",
    kind: str = "note",
    tags: Iterable[str] = (),
    source: str = "input",
    origin: str = "",
) -> Document | None:
    """Wrap arbitrary user input as a Document the pipeline understands."""
    text = clean_text(text)
    if len(text) < 40:
        log.warning("input is too short to learn anything from (%d chars)", len(text))
        return None

    if not title:
        # First non-empty line, stripped of markdown heading marks.
        first = next((l.strip(" #*-") for l in text.split("\n") if l.strip()), "")
        title = truncate(first, 80) or f"input {now_iso()}"

    return Document.make(
        title=title,
        url=origin or f"input://{kind}/{_slug(title)}-{short_id(text, length=8)}",
        text=text,
        kind=kind,
        source=source,
        tags=list(tags) + ["user-input"],
        meta={"entered_at": now_iso(), "chars": len(text)},
    )


def pair_record(
    question: str,
    answer: str,
    *,
    kind: str = "qa",
    tags: Iterable[str] = (),
    context: str = "",
    title: str = "",
) -> Record:
    """A Q&A pair written by a person.

    Deliberately bypasses generation AND the quality gate. The gate exists to
    catch a small model hallucinating; applying it to a human-authored
    correction would be backwards — its grounding check would reject a true
    answer simply because you did not paste the source it came from.
    """
    question, answer = clean_text(question), clean_text(answer)
    title = title or truncate(question, 70)
    url = f"human://{kind}/{_slug(question)}-{short_id(question, answer, length=8)}"

    doc = Document.make(
        title=title,
        url=url,
        text=f"Q: {question}\n\nA: {answer}",
        kind=kind,
        source="human",
        tags=list(tags) + ["user-input", "human-authored"],
    )
    chunk = Chunk.make(doc, context or doc.text, 0)

    rec = Record.make(
        kind, question, answer, chunk,
        generator="human:input",
        meta={"authored": now_iso()},
    )
    rec.score = 1.0
    rec.score_reason = "human-authored, trusted without judging"
    rec.tags = list(dict.fromkeys(rec.tags + ["human-authored"]))
    return rec


def case_record(problem: str, resolution: str, *, tags: Iterable[str] = ()) -> Record:
    """A solved case: something broke, here is what fixed it."""
    return pair_record(
        f"I am seeing this problem:\n{clean_text(problem)}\n\nWhat is the cause and how do I fix it?",
        clean_text(resolution),
        kind="troubleshooting",
        tags=list(tags) + ["solved-case"],
        title=truncate(f"Case: {problem}", 70),
    )


def collect_input(
    cfg: Config,
    *,
    text: str = "",
    files: Iterable[str] = (),
    urls: Iterable[str] = (),
    kind: str = "note",
    title: str = "",
    tags: Iterable[str] = (),
) -> list[Document]:
    """Gather every provided input into Documents.

    Sources are additive: `learn --file a.md --url http://… "extra note"` folds
    all three into one batch.
    """
    docs: list[Document] = []
    tags = list(tags)

    if text.strip():
        doc = document_from_text(text, title=title, kind=kind, tags=tags)
        if doc:
            docs.append(doc)

    for raw in files:
        path = Path(raw).expanduser()
        if not path.exists():
            log.error("no such file: %s", path)
            continue
        if path.is_dir():
            from .connectors.files import iter_files, read_file

            for child in iter_files(path, []):
                doc = read_file(child, "input", tags + ["user-input"])
                if doc:
                    docs.append(doc)
            continue

        from .connectors.files import read_file

        doc = read_file(path, "input", tags + ["user-input"])
        if doc:
            # Runbook-shaped input gets the structural parser.
            if kind == "runbook":
                from .connectors.runbooks import normalize, parse_runbook

                doc.text = normalize(parse_runbook(doc.text, doc.title), doc.title)
                doc.kind = "runbook"
            docs.append(doc)
        else:
            log.error("could not read anything usable from %s", path)

    for url in urls:
        from .connectors.web import fetch_url

        doc = fetch_url(url, source="input", tags=tags + ["user-input"])
        if doc:
            docs.append(doc)
        else:
            log.error("nothing usable at %s", url)

    return docs
