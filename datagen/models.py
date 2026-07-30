"""Core data types that flow through the pipeline.

    Source  ──fetch──▶  Document  ──chunk──▶  Chunk  ──generate──▶  Record
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from .util import now_iso, short_id, tokenize


@dataclass
class Document:
    """One retrieved artefact: a PDF, a Confluence page, a crawled URL, a runbook."""

    id: str
    title: str
    url: str                       # real URL, or file:// / confluence:// / runbook://
    text: str
    kind: str                      # pdf | html | markdown | confluence | runbook | docx …
    source: str                    # the configured source name it came from
    tags: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=now_iso)
    content_hash: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make(title: str, url: str, text: str, kind: str, source: str, **kw: Any) -> "Document":
        from .util import sha256

        return Document(
            id=short_id(url, title),
            title=title.strip() or url,
            url=url,
            text=text,
            kind=kind,
            source=source,
            content_hash=sha256(text),
            **kw,
        )


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    url: str
    text: str
    index: int                     # position within the document
    kind: str
    source: str
    tags: list[str] = field(default_factory=list)
    heading: str = ""              # nearest enclosing heading, when detectable
    tokens: list[str] = field(default_factory=list)
    embedding: list[float] | None = None

    @staticmethod
    def make(doc: Document, text: str, index: int, heading: str = "") -> "Chunk":
        return Chunk(
            id=short_id(doc.id, index, text[:64]),
            doc_id=doc.id,
            title=doc.title,
            url=doc.url,
            text=text,
            index=index,
            kind=doc.kind,
            source=doc.source,
            tags=list(doc.tags),
            heading=heading,
            tokens=tokenize(text),
        )


@dataclass
class Record:
    """One dataset row: an instruction/response pair with provenance."""

    id: str
    kind: str                      # qa | instruction | troubleshooting | summary
    instruction: str               # the question / task
    output: str                    # the answer
    input: str = ""                # optional extra context (alpaca `input`)
    context: str = ""              # the source chunk text, for grounding checks
    source_url: str = ""
    source_title: str = ""
    chunk_id: str = ""
    tags: list[str] = field(default_factory=list)
    score: float | None = None     # quality-gate score, 0..1
    score_reason: str = ""
    quarantined: bool = False
    generator: str = ""            # llm:<model> or extractive:<strategy>
    created_at: str = field(default_factory=now_iso)

    @staticmethod
    def make(kind: str, instruction: str, output: str, chunk: Chunk, **kw: Any) -> "Record":
        return Record(
            id=short_id(chunk.id, kind, instruction),
            kind=kind,
            instruction=instruction.strip(),
            output=output.strip(),
            context=chunk.text,
            source_url=chunk.url,
            source_title=chunk.title,
            chunk_id=chunk.id,
            tags=list(chunk.tags),
            **kw,
        )

    def to_dict(self, include_context: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_context:
            d.pop("context", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class RunStats:
    """Tallies for one pipeline run — printed at the end and stored in state."""

    run_id: str = ""
    started_at: str = field(default_factory=now_iso)
    finished_at: str = ""
    docs_fetched: int = 0
    docs_skipped_unchanged: int = 0
    chunks: int = 0
    chunks_new: int = 0
    duplicates_exact: int = 0
    duplicates_near: int = 0
    duplicates_semantic: int = 0
    records: int = 0
    records_quarantined: int = 0
    llm_calls: int = 0
    llm_failures: int = 0
    extractive_fallbacks: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"docs={self.docs_fetched} (skipped {self.docs_skipped_unchanged}) "
            f"chunks={self.chunks} (new {self.chunks_new}) "
            f"dupes={self.duplicates_exact + self.duplicates_near + self.duplicates_semantic} "
            f"records={self.records} (quarantined {self.records_quarantined}) "
            f"llm={self.llm_calls} fail={self.llm_failures} "
            f"fallback={self.extractive_fallbacks} errors={len(self.errors)}"
        )
