"""Source connectors.

A connector turns a config block into `Document`s. Adding a new source type is
a matter of writing `fetch(cfg, block, state) -> list[Document]` and registering
it here — the pipeline and the agent both discover connectors through
`CONNECTORS`.
"""

from __future__ import annotations

from typing import Any, Callable

from ..models import Document
from . import confluence, files, runbooks, search, web

# name in config `[[sources.<name>]]` -> fetch function
CONNECTORS: dict[str, Callable[..., list[Document]]] = {
    "files": files.fetch,
    "runbooks": runbooks.fetch,
    "confluence": confluence.fetch,
    "web": web.fetch,
    "keywords": search.fetch,
}


def source_blocks(sources: dict[str, Any]) -> list[tuple[str, dict]]:
    """Normalise the config's sources table into (type, block) pairs.

    `[[sources.web]]` yields a list of blocks; `[sources.keywords]` yields one.
    """
    out: list[tuple[str, dict]] = []
    for type_, value in (sources or {}).items():
        if type_ not in CONNECTORS:
            continue
        blocks = value if isinstance(value, list) else [value]
        for block in blocks:
            if isinstance(block, dict) and block.get("enabled", True):
                out.append((type_, block))
    return out


__all__ = ["CONNECTORS", "source_blocks", "files", "runbooks", "confluence", "web", "search"]
