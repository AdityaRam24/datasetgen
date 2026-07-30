"""Web connector: a polite, bounded, breadth-first crawler.

Design constraints that matter:
  * robots.txt is honoured by default (`respect_robots`)
  * one domain allow-list per source block; off-list links are never fetched
  * a fixed delay between requests to the same host
  * hard caps on pages, depth and response size
  * binary responses (PDF!) are parsed, not discarded — vendor docs are full of
    linked PDFs and those are often the most useful pages

The crawler is also used by the agent as a tool (`scrape_url`), which is why
`fetch_url` is exposed separately from `crawl`.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.robotparser
from typing import Any, Iterable
from urllib.parse import urlparse

from ..models import Document
from ..util import HttpError, get_logger, http_request
from .parsers import html_links, parse_bytes

log = get_logger("web")

BINARY_KINDS = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "text/markdown": "markdown",
    "text/plain": "text",
    "application/json": "json",
    "text/csv": "csv",
}


def _extra_headers() -> dict[str, str]:
    raw = os.getenv("DATAGEN_WEB_HEADERS", "").strip()
    if not raw:
        return {}
    try:
        return {str(k): str(v) for k, v in json.loads(raw).items()}
    except (ValueError, AttributeError):
        log.warning("DATAGEN_WEB_HEADERS is not valid JSON — ignoring")
        return {}


class RobotsCache:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str, agent: str = "*") -> bool:
        if not self.enabled:
            return True
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            try:
                resp = http_request(f"{origin}/robots.txt", timeout=10, retries=0)
                rp.parse(resp.text.splitlines())
                self._cache[origin] = rp
            except HttpError:
                self._cache[origin] = None  # no robots.txt == crawl allowed
        rp = self._cache[origin]
        if rp is None:
            return True
        try:
            return rp.can_fetch(agent, url)
        except Exception:
            return True


def url_allowed(url: str, allow_domains: list[str], deny_patterns: list[str]) -> bool:
    host = urlparse(url).netloc.lower().split(":")[0]
    if allow_domains and not any(host == d or host.endswith("." + d) for d in allow_domains):
        return False
    return not any(re.search(p, url, re.I) for p in deny_patterns or [])


def fetch_url(url: str, timeout: float = 30.0, source: str = "web", tags: Iterable[str] = ()) -> Document | None:
    """Fetch and parse a single URL into a Document. Returns None on failure or
    when the response holds no usable text."""
    try:
        resp = http_request(url, timeout=timeout, headers=_extra_headers())
    except HttpError as e:
        log.warning("fetch failed: %s", e)
        return None

    ctype = resp.content_type
    kind = BINARY_KINDS.get(ctype, "html" if "html" in ctype or not ctype else None)
    if kind is None:
        # Fall back to the extension when the server sends a useless MIME type.
        ext = os.path.splitext(urlparse(url).path)[1].lower()
        from .parsers import EXTENSION_MAP

        kind = EXTENSION_MAP.get(ext)
        if kind is None:
            log.debug("skipping %s (content-type %s)", url, ctype)
            return None

    try:
        title, text = parse_bytes(resp.body, kind, url=resp.url)
    except Exception as e:
        log.warning("parse failed for %s: %s", url, e)
        return None

    if not text or len(text) < 200:
        log.debug("too little text at %s (%d chars)", url, len(text or ""))
        return None

    return Document.make(
        title=title or urlparse(url).path.rsplit("/", 1)[-1] or url,
        url=resp.url,
        text=text,
        kind=kind,
        source=source,
        tags=list(tags),
        meta={"content_type": ctype, "status": resp.status, "bytes": len(resp.body)},
    )


def crawl(
    seeds: list[str],
    *,
    allow_domains: list[str] | None = None,
    deny_patterns: list[str] | None = None,
    max_pages: int = 50,
    max_depth: int = 2,
    delay_ms: int = 800,
    respect_robots: bool = True,
    source: str = "web",
    tags: Iterable[str] = (),
    skip_urls: set[str] | None = None,
) -> list[Document]:
    """Breadth-first crawl from `seeds`. Returns the parsed documents."""
    allow_domains = allow_domains or [urlparse(s).netloc for s in seeds if s]
    deny_patterns = deny_patterns or []
    robots = RobotsCache(respect_robots)
    skip_urls = skip_urls or set()

    queue: list[tuple[str, int]] = [(s, 0) for s in seeds]
    seen: set[str] = set(seeds)
    docs: list[Document] = []
    last_hit: dict[str, float] = {}

    while queue and len(docs) < max_pages:
        url, depth = queue.pop(0)

        if not url_allowed(url, allow_domains, deny_patterns):
            continue
        if not robots.allowed(url):
            log.debug("robots.txt disallows %s", url)
            continue

        # Per-host politeness delay.
        host = urlparse(url).netloc
        wait = (delay_ms / 1000.0) - (time.time() - last_hit.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        last_hit[host] = time.time()

        if url in skip_urls:
            log.debug("already ingested, still following links: %s", url)
        else:
            doc = fetch_url(url, source=source, tags=tags)
            if doc:
                docs.append(doc)
                log.info("  [%d/%d] %s", len(docs), max_pages, doc.title[:70])

        if depth >= max_depth:
            continue

        # Re-fetch cheaply for links only when the page was HTML.
        try:
            resp = http_request(url, timeout=20, retries=0, headers=_extra_headers())
            if "html" not in resp.content_type:
                continue
            for link in html_links(resp.text, resp.url):
                if link in seen or len(seen) > max_pages * 12:
                    continue
                if not url_allowed(link, allow_domains, deny_patterns):
                    continue
                seen.add(link)
                queue.append((link, depth + 1))
        except HttpError:
            continue

    log.info("crawl finished: %d documents from %d seeds", len(docs), len(seeds))
    return docs


def fetch(cfg: Any, block: dict, state: Any = None) -> list[Document]:
    name = block.get("name", "web")
    seeds = list(block.get("seeds", []))

    # Pull in URLs the agent proposed on earlier runs — this is one half of the
    # "updates itself" behaviour.
    if state is not None:
        for row in state.pending("url", limit=25):
            if url_allowed(row["key"], block.get("allow_domains", []), block.get("deny_patterns", [])):
                seeds.append(row["key"])
                state.mark(row["key"], "accepted")

    if not seeds:
        log.info("[%s] no seeds configured", name)
        return []

    log.info("[%s] crawling %d seeds (max %s pages, depth %s)",
             name, len(seeds), block.get("max_pages", 50), block.get("max_depth", 2))

    return crawl(
        seeds,
        allow_domains=block.get("allow_domains", []),
        deny_patterns=block.get("deny_patterns", []),
        max_pages=int(block.get("max_pages", 50)),
        max_depth=int(block.get("max_depth", 2)),
        delay_ms=int(block.get("delay_ms", 800)),
        respect_robots=bool(block.get("respect_robots", True)),
        source=name,
        tags=block.get("tags", []),
    )
