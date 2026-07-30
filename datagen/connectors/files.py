"""Local file connector: PDFs, Office documents, Markdown, text, code.

Walks the configured directory, parses whatever it can, and emits one Document
per file. Files are skipped when their mtime+size fingerprint matches the last
run, so re-running over a large corpus is cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models import Document
from ..util import get_logger, human_bytes, sha256
from .parsers import EXTENSION_MAP, parse_bytes

log = get_logger("files")

MAX_FILE_BYTES = 80 * 1024 * 1024
DEFAULT_GLOBS = ["**/*.pdf", "**/*.docx", "**/*.pptx", "**/*.xlsx", "**/*.md", "**/*.txt", "**/*.html"]
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".idea"}


def iter_files(root: Path, globs: list[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in globs or DEFAULT_GLOBS:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(path)
    return out


def read_file(path: Path, source: str, tags: list[str] | None = None) -> Document | None:
    """Parse one file into a Document, or None if it is unreadable/empty."""
    # as_uri() below requires an absolute path. The connector's own globbing
    # always produces one, but `datagen learn --file ./notes.md` and
    # `datagen ingest ./docs` hand us whatever the user typed.
    path = path.expanduser().resolve()
    kind = EXTENSION_MAP.get(path.suffix.lower())
    if kind is None:
        log.debug("skipping unsupported file type: %s", path.name)
        return None

    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            log.warning("skipping %s (%s > limit)", path.name, human_bytes(size))
            return None
        data = path.read_bytes()
    except OSError as e:
        log.warning("cannot read %s: %s", path, e)
        return None

    try:
        title, text = parse_bytes(data, kind, url=path.as_uri())
    except Exception as e:  # a corrupt file must not abort the whole run
        log.warning("failed to parse %s: %s", path.name, e)
        return None

    if not text or len(text) < 40:
        log.debug("no usable text in %s", path.name)
        return None

    return Document.make(
        title=title or path.stem.replace("_", " ").replace("-", " "),
        url=path.as_uri(),
        text=text,
        kind=kind,
        source=source,
        tags=list(tags or []),
        meta={"path": str(path), "bytes": size, "fingerprint": _fingerprint(path)},
    )


def _fingerprint(path: Path) -> str:
    try:
        st = path.stat()
        return sha256(f"{path}:{st.st_size}:{int(st.st_mtime)}")[:16]
    except OSError:
        return ""


def fetch(cfg: Any, block: dict, state: Any = None) -> list[Document]:
    name = block.get("name", "files")
    root = cfg.resolve(block.get("path", "corpus"))
    if not root.exists():
        log.warning("[%s] path does not exist: %s — create it and drop documents in", name, root)
        return []

    paths = iter_files(root, block.get("globs", DEFAULT_GLOBS))
    log.info("[%s] %d candidate files under %s", name, len(paths), root)

    docs: list[Document] = []
    for path in paths:
        doc = read_file(path, name, block.get("tags", []))
        if doc:
            docs.append(doc)
    log.info("[%s] parsed %d documents", name, len(docs))
    return docs
