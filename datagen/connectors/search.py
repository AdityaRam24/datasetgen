"""Keyword -> web discovery.

"Scrape everything you can find on this keyword" is implemented here: a keyword
goes to a search engine, the result URLs are ranked, and the promising ones are
scraped through the normal web connector.

Engines:
  * duckduckgo — the no-JS HTML endpoint, no API key needed
  * searxng    — a self-hosted SearXNG instance (`searxng_url`), JSON API

If both fail (offline, rate-limited), the keyword is recorded in state so a
later run retries it, and the run continues on the other sources.
"""

from __future__ import annotations

import html
import re
import time
from typing import Any
from urllib.parse import parse_qs, quote_plus, urlparse

from ..models import Document
from ..util import HttpError, get_logger, http_request
from .web import fetch_url, url_allowed

log = get_logger("search")

# Domains that are almost never worth training on.
JUNK_DOMAINS = {
    "pinterest.com", "facebook.com", "x.com", "twitter.com", "instagram.com",
    "quora.com", "slideshare.net", "scribd.com", "coursehero.com",
}
# Domains that usually carry primary/authoritative technical material.
PREFERRED_HINTS = (
    "docs.", "developer.", "documentation", "/docs/", "support.", "kb.",
    "github.com", "gitlab.com", "readthedocs", "man7.org", "kubernetes.io",
)


class SearchResult:
    __slots__ = ("url", "title", "snippet", "rank", "score")

    def __init__(self, url: str, title: str = "", snippet: str = "", rank: int = 0) -> None:
        self.url, self.title, self.snippet, self.rank = url, title, snippet, rank
        self.score = 0.0

    def __repr__(self) -> str:
        return f"<SearchResult {self.score:.2f} {self.url}>"


def search_duckduckgo(query: str, limit: int = 10) -> list[SearchResult]:
    """DuckDuckGo's HTML endpoint. No key, no JS; result markup is simple."""
    try:
        resp = http_request(
            f"https://html.duckduckgo.com/html/?q={quote_plus(query)}",
            headers={
                "Accept": "text/html",
                # The HTML endpoint serves a stripped page to plain UAs.
                "User-Agent": "Mozilla/5.0 (compatible; kalam-datagen/1.0)",
            },
            timeout=25,
            retries=1,
        )
    except HttpError as e:
        log.warning("duckduckgo search failed for %r: %s", query, e)
        return []

    results: list[SearchResult] = []
    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
        r'(?:.*?<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>)?',
        re.S | re.I,
    )
    for i, m in enumerate(pattern.finditer(resp.text)):
        url = _unwrap_ddg(html.unescape(m.group(1)))
        if not url:
            continue
        results.append(
            SearchResult(
                url=url,
                title=_strip_tags(m.group(2) or ""),
                snippet=_strip_tags(m.group(3) or ""),
                rank=i,
            )
        )
        if len(results) >= limit:
            break
    return results


def _unwrap_ddg(href: str) -> str:
    """DDG wraps results as //duckduckgo.com/l/?uddg=<encoded>."""
    if "duckduckgo.com/l/" in href or href.startswith("/l/"):
        qs = parse_qs(urlparse(href if href.startswith("http") else "https:" + href).query)
        target = (qs.get("uddg") or [""])[0]
        return target or ""
    return href if href.startswith("http") else ""


def _strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


def search_searxng(query: str, base_url: str, limit: int = 10) -> list[SearchResult]:
    """Query a local SearXNG instance's JSON API.

    SearXNG aggregates Google/Bing/DDG/StackOverflow at once, so it gives
    better coverage than any single engine and does not rate-limit you — but
    only if `search.formats` includes `json` in its settings.yml. Without it
    every request is a bare 403, which is why that case is called out here.
    """
    try:
        data = http_request(
            f"{base_url.rstrip('/')}/search?q={quote_plus(query)}&format=json",
            timeout=30,
            retries=1,
        ).json()
    except HttpError as e:
        if e.status == 403:
            log.error(
                "SearXNG returned 403 — its JSON API is disabled. Add `json` to "
                "`search.formats` and set `server.limiter: false` in "
                "searxng/settings.yml, then: python searxng/setup.py --restart"
            )
        else:
            log.warning("searxng search failed for %r: %s", query, e)
        return []
    except ValueError as e:
        log.warning("searxng returned invalid JSON for %r: %s", query, e)
        return []

    return [
        SearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            snippet=item.get("content", ""),
            rank=i,
        )
        for i, item in enumerate(data.get("results", [])[:limit])
        if item.get("url")
    ]


def searxng_available(base_url: str) -> bool:
    if not base_url:
        return False
    try:
        http_request(f"{base_url.rstrip('/')}/healthz", timeout=5, retries=0)
        return True
    except HttpError:
        return False


def web_search(
    query: str, engine: str = "searxng", searxng_url: str = "", limit: int = 10
) -> list[SearchResult]:
    """Search, preferring the configured engine but never dead-ending.

    SearXNG is the better source when it is up: it aggregates several engines
    and is not scraping anyone's HTML. When the container is stopped we fall
    back to DuckDuckGo rather than failing the run — a paused container should
    not cost you a build.
    """
    results: list[SearchResult] = []

    if engine == "searxng" and searxng_url:
        results = search_searxng(query, searxng_url, limit)
        if not results:
            if not searxng_available(searxng_url):
                log.warning(
                    "SearXNG at %s is not reachable — falling back to DuckDuckGo. "
                    "Start it with: python searxng/setup.py",
                    searxng_url,
                )
            results = search_duckduckgo(query, limit)
    else:
        results = search_duckduckgo(query, limit)

    return rank_results(results, query)


def rank_results(results: list[SearchResult], query: str) -> list[SearchResult]:
    """Heuristic ranking: prefer primary docs, penalise junk and dead weight.

    The agent can override this with a local-LLM judgement, but the heuristic
    alone is good enough to keep obvious rubbish out of the dataset.
    """
    terms = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}

    ranked: list[SearchResult] = []
    for r in results:
        host = urlparse(r.url).netloc.lower().removeprefix("www.")
        if any(host == d or host.endswith("." + d) for d in JUNK_DOMAINS):
            continue

        score = 1.0 - (r.rank * 0.03)                 # search-engine order
        if any(h in r.url.lower() for h in PREFERRED_HINTS):
            score += 0.45
        haystack = f"{r.title} {r.snippet}".lower()
        overlap = sum(1 for t in terms if t in haystack)
        score += 0.35 * (overlap / max(1, len(terms)))
        if r.url.lower().endswith(".pdf"):
            score += 0.15                              # vendor PDFs are gold
        if re.search(r"/(tag|category|archive|page)/\d*", r.url):
            score -= 0.4
        r.score = round(score, 3)
        ranked.append(r)

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked


def scrape_keyword(
    keyword: str,
    *,
    engine: str = "duckduckgo",
    searxng_url: str = "",
    results_per_keyword: int = 8,
    max_pages: int = 10,
    source: str = "keywords",
    tags: list[str] | None = None,
    allow_domains: list[str] | None = None,
    deny_patterns: list[str] | None = None,
    min_score: float = 0.0,
    delay_ms: int = 1000,
) -> list[Document]:
    """Search for a keyword and scrape the best results."""
    results = web_search(keyword, engine, searxng_url, results_per_keyword)
    if not results:
        return []

    log.info("[keyword] %r -> %d results", keyword, len(results))
    docs: list[Document] = []
    for r in results:
        if len(docs) >= max_pages:
            break
        if r.score < min_score:
            continue
        if allow_domains and not url_allowed(r.url, allow_domains, deny_patterns or []):
            continue
        doc = fetch_url(r.url, source=source, tags=(tags or []) + ["keyword:" + keyword])
        if doc:
            doc.meta["search_rank"] = r.rank
            doc.meta["search_score"] = r.score
            doc.meta["keyword"] = keyword
            docs.append(doc)
            log.info("    + %s (%.2f)", doc.title[:60], r.score)
        time.sleep(delay_ms / 1000.0)
    return docs


def fetch(cfg: Any, block: dict, state: Any = None) -> list[Document]:
    terms = list(block.get("terms", []))

    # Keywords the agent proposed on earlier runs — the self-expanding half.
    if state is not None:
        for row in state.pending("keyword", limit=15):
            terms.append(row["key"])
            state.mark(row["key"], "accepted")

    if not terms:
        return []

    docs: list[Document] = []
    for term in terms:
        docs.extend(
            scrape_keyword(
                term,
                engine=block.get("engine", "duckduckgo"),
                searxng_url=block.get("searxng_url", ""),
                results_per_keyword=int(block.get("results_per_keyword", 8)),
                max_pages=int(block.get("max_pages_per_keyword", 10)),
                source=block.get("name", "keywords"),
                tags=block.get("tags", ["keyword"]),
                allow_domains=block.get("allow_domains"),
                deny_patterns=block.get("deny_patterns"),
            )
        )
    log.info("[keywords] %d documents from %d terms", len(docs), len(terms))
    return docs
