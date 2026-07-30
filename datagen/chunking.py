"""Heading-aware chunking.

Two strategies:
  * structured — splits on Markdown/ATX headings and keeps the heading trail on
    each chunk, so "Step 3" in a runbook still knows which procedure it belongs
    to. This matters a lot for generated Q&A quality.
  * sliding    — paragraph-packing with overlap, for unstructured text (PDF).

The output is compatible with the chunker in `server/pcai/store.ts`, so chunks
exported into the Kalam KB behave the same as natively ingested ones.
"""

from __future__ import annotations

import re

from .config import ChunkingConfig
from .models import Chunk, Document
from .util import clean_text, get_logger

log = get_logger("chunking")

# Markdown ATX headings, setext underlines, and ALL-CAPS section labels that
# PDF/runbook text tends to use.
_HEADING = re.compile(
    r"^(?:(#{1,6})\s+(?P<atx>.+?)\s*#*|"
    r"(?P<caps>[A-Z][A-Z0-9 _\-/&()]{3,60})|"
    r"(?P<num>\d+(?:\.\d+)*\.?\s+[A-Z].{2,70}))$"
)
_SETEXT = re.compile(r"^(=|-){3,}\s*$")


def _heading_level(line: str, next_line: str | None) -> tuple[int, str] | None:
    line = line.strip()
    if not line or len(line) > 120:
        return None
    if next_line is not None and _SETEXT.match(next_line.strip()):
        return (1 if next_line.strip()[0] == "=" else 2, line)
    m = _HEADING.match(line)
    if not m:
        return None
    if m.group("atx"):
        return len(m.group(1)), m.group("atx").strip()
    if m.group("caps"):
        return 2, m.group("caps").strip()
    if m.group("num"):
        depth = m.group("num").split()[0].count(".") + 1
        return min(depth, 6), m.group("num").strip()
    return None


def _sections(text: str) -> list[tuple[str, str]]:
    """Split into (heading_trail, body) pairs."""
    lines = text.split("\n")
    trail: list[str] = []
    sections: list[tuple[str, str]] = []
    buf: list[str] = []
    current = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        head = _heading_level(line, nxt)
        if head:
            level, title = head
            if buf and any(b.strip() for b in buf):
                sections.append((current, "\n".join(buf).strip()))
            buf = []
            trail = trail[: level - 1]
            trail.append(title)
            current = " › ".join(trail)
            if nxt is not None and _SETEXT.match(nxt.strip()):
                i += 1  # consume the underline
        else:
            buf.append(line)
        i += 1

    if buf and any(b.strip() for b in buf):
        sections.append((current, "\n".join(buf).strip()))
    return sections or [("", text)]


def _pack(text: str, max_chars: int, overlap: int) -> list[str]:
    """Pack paragraphs up to max_chars, carrying `overlap` chars of tail into
    the next chunk so a fact split across a boundary is still recoverable."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    for p in paras:
        # A single monster paragraph (common in PDFs): hard-split on sentences.
        if len(p) > max_chars:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            for piece in _split_sentences(p, max_chars, overlap):
                chunks.append(piece)
            continue
        if buf and len(buf) + len(p) + 2 > max_chars:
            chunks.append(buf.strip())
            buf = (buf[-overlap:] + "\n\n" + p) if overlap else p
        else:
            buf = f"{buf}\n\n{p}" if buf else p

    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _split_sentences(text: str, max_chars: int, overlap: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    buf = ""
    for s in sentences:
        while len(s) > max_chars:            # no sentence breaks at all
            out.append(s[:max_chars])
            s = s[max_chars - overlap :]
        if buf and len(buf) + len(s) + 1 > max_chars:
            out.append(buf.strip())
            buf = (buf[-overlap:] + " " + s) if overlap else s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    text = clean_text(doc.text)
    if not text:
        return []

    pieces: list[tuple[str, str]] = []  # (heading, body)
    if cfg.respect_headings:
        for heading, body in _sections(text):
            for part in _pack(body, cfg.max_chars, cfg.overlap):
                pieces.append((heading, part))
    else:
        pieces = [("", p) for p in _pack(text, cfg.max_chars, cfg.overlap)]

    chunks: list[Chunk] = []
    for body_index, (heading, body) in enumerate(pieces):
        if len(body) < cfg.min_chars:
            # Keep a short section only if it is the document's entire content.
            if len(pieces) > 1:
                continue
        # Prefix the heading trail so the chunk is self-describing to the LLM.
        text_out = f"{heading}\n\n{body}" if heading and heading not in body else body
        chunks.append(Chunk.make(doc, text_out, len(chunks), heading))
        del body_index

    log.debug("%s -> %d chunks", doc.title[:50], len(chunks))
    return chunks
