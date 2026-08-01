"""Confluence connector — Cloud and Server/Data Center.

Auth (from the environment, never the config file):
  * Cloud: CONFLUENCE_USER + CONFLUENCE_TOKEN -> HTTP Basic
  * Server/DC: CONFLUENCE_TOKEN only -> Bearer personal access token

Pages come back as storage-format XHTML, which is cleaned into readable text;
Confluence macros (code blocks, info/warning panels, status labels, expands)
are unwrapped rather than dropped, because in an ops space the code macro is
usually the part you actually want.

Attachments (PDF/DOCX/PPTX) are optionally pulled and run through the same
parsers as local files.
"""

from __future__ import annotations

import base64
import html
import os
import re
from typing import Any, Iterator
from urllib.parse import quote, urlparse

from ..models import Document
from ..util import HttpError, clean_text, get_logger, http_request
from .parsers import EXTENSION_MAP, parse_bytes, parse_html

log = get_logger("confluence")


class ConfluenceClient:
    def __init__(self, base_url: str = "", user: str = "", token: str = "") -> None:
        self.base_url = (base_url or os.getenv("CONFLUENCE_BASE_URL", "")).rstrip("/")
        self.user = user or os.getenv("CONFLUENCE_USER", "")
        self.token = token or os.getenv("CONFLUENCE_TOKEN", "")
        self._cloud: bool | None = None

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    @property
    def headers(self) -> dict[str, str]:
        h = {"Accept": "application/json"}
        if self.user:
            raw = f"{self.user}:{self.token}".encode()
            h["Authorization"] = "Basic " + base64.b64encode(raw).decode()
        elif self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def ping(self) -> bool:
        if not self.configured:
            return False
        try:
            http_request(f"{self.base_url}/rest/api/space?limit=1",
                         headers=self.headers, timeout=20, retries=0)
            return True
        except HttpError as e:
            log.warning("Confluence not reachable: %s", e)
            return False

    def search(self, cql: str, limit: int = 500, expand: str = "body.storage,version,space,metadata.labels") -> Iterator[dict]:
        """Page through /rest/api/content/search."""
        start, fetched = 0, 0
        page_size = min(50, limit)
        while fetched < limit:
            url = (
                f"{self.base_url}/rest/api/content/search"
                f"?cql={quote(cql)}&limit={page_size}&start={start}&expand={quote(expand)}"
            )
            try:
                data = http_request(url, headers=self.headers, timeout=45, retries=1).json()
            except (HttpError, ValueError) as e:
                log.warning("Confluence search failed: %s", e)
                return
            results = data.get("results", [])
            if not results:
                return
            for item in results:
                yield item
                fetched += 1
                if fetched >= limit:
                    return
            start += len(results)
            if not data.get("_links", {}).get("next"):
                return

    def spaces(self, limit: int = 200) -> list[dict]:
        """Every space this token can see: [{key, name}, ...].

        Used to check the configured keys before searching. A CQL query naming a
        space that does not exist returns an empty result set with a 200, which
        is indistinguishable from "the wiki has nothing" unless you look.
        """
        out: list[dict] = []
        start = 0
        while len(out) < limit:
            url = f"{self.base_url}/rest/api/space?limit=50&start={start}"
            try:
                data = http_request(url, headers=self.headers, timeout=30, retries=0).json()
            except (HttpError, ValueError) as e:
                log.debug("space listing failed: %s", e)
                break
            results = data.get("results", [])
            if not results:
                break
            out.extend({"key": s.get("key", ""), "name": s.get("name", "")} for s in results)
            start += len(results)
            if not (data.get("_links") or {}).get("next"):
                break
        return out

    def attachments(self, page_id: str, limit: int = 25) -> list[dict]:
        url = f"{self.base_url}/rest/api/content/{page_id}/child/attachment?limit={limit}"
        try:
            return http_request(url, headers=self.headers, timeout=30, retries=0).json().get("results", [])
        except (HttpError, ValueError):
            return []

    def download(self, download_path: str) -> bytes | None:
        origin = f"{urlparse(self.base_url).scheme}://{urlparse(self.base_url).netloc}"
        url = download_path if download_path.startswith("http") else origin + download_path
        try:
            return http_request(url, headers=self.headers, timeout=60, retries=1).body
        except HttpError as e:
            log.debug("attachment download failed: %s", e)
            return None


# ---------------------------------------------------------------------------
# Storage-format cleanup
# ---------------------------------------------------------------------------

_CODE_MACRO = re.compile(
    r'<ac:structured-macro[^>]*ac:name="(code|noformat)".*?'
    r'<ac:plain-text-body><!\[CDATA\[(.*?)\]\]></ac:plain-text-body>.*?</ac:structured-macro>',
    re.S | re.I,
)
_PANEL_MACRO = re.compile(
    r'<ac:structured-macro[^>]*ac:name="(info|note|warning|tip|panel|expand)"[^>]*>(.*?)</ac:structured-macro>',
    re.S | re.I,
)
_STATUS_MACRO = re.compile(
    r'<ac:structured-macro[^>]*ac:name="status".*?<ac:parameter[^>]*ac:name="title">(.*?)</ac:parameter>.*?</ac:structured-macro>',
    re.S | re.I,
)
_ANY_MACRO = re.compile(r"<ac:structured-macro.*?</ac:structured-macro>", re.S | re.I)
_LINK = re.compile(r'<ac:link[^>]*>.*?<ri:page[^>]*ri:content-title="([^"]*)".*?</ac:link>', re.S | re.I)
_CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.S)


def storage_to_text(storage: str) -> str:
    """Confluence storage XHTML -> markdown-ish plain text."""
    if not storage:
        return ""
    text = storage

    text = _CODE_MACRO.sub(lambda m: f"\n\n```\n{m.group(2).strip()}\n```\n\n", text)
    text = _STATUS_MACRO.sub(lambda m: f"[{_strip(m.group(1))}]", text)
    text = _PANEL_MACRO.sub(lambda m: f"\n\n> **{m.group(1).upper()}:** {m.group(2)}\n\n", text)
    text = _LINK.sub(lambda m: f"[{m.group(1)}]", text)
    text = _ANY_MACRO.sub(" ", text)              # remaining macros: drop
    text = _CDATA.sub(r"\1", text)
    text = re.sub(r"<ac:[^>]+>|</ac:[^>]+>|<ri:[^>]+/?>", " ", text)

    _, body = parse_html(f"<html><body>{text}</body></html>")
    return clean_text(body)


def _strip(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text)).strip()


# ---------------------------------------------------------------------------


def build_cql(block: dict, spaces: list[str] | None = None, text_terms: list[str] | None = None) -> str:
    if block.get("cql"):
        return str(block["cql"])
    spaces = block.get("spaces", []) if spaces is None else spaces
    parts = ["type = page"]
    if spaces:
        joined = ", ".join(f'"{s}"' for s in spaces)
        parts.append(f"space in ({joined})")
    if text_terms:
        ors = " OR ".join(f'text ~ "{t}"' for t in text_terms if t)
        if ors:
            parts.append(f"({ors})")
    if block.get("labels"):
        labels = ", ".join(f'"{l}"' for l in block["labels"])
        parts.append(f"label in ({labels})")
    if block.get("updated_since"):        # e.g. "-30d" or "2026-01-01"
        parts.append(f'lastmodified >= "{block["updated_since"]}"')
    return " AND ".join(parts) + " ORDER BY lastmodified DESC"


def _resolve_cql(client: ConfluenceClient, block: dict, name: str, terms: list[str]) -> str:
    """Pick a CQL query that will actually match something.

    Configured space keys are checked against the wiki first: a key that does
    not exist matches zero pages and reports no error, which is exactly how a
    run ends with `fetched: 0` and no explanation. When none of the configured
    keys exist we fall back to a text search across every readable space, which
    is what you want from a wiki you have access to but have not catalogued.
    """
    if block.get("cql"):
        return str(block["cql"])

    wanted = [str(s).strip() for s in block.get("spaces", []) if str(s).strip()]
    if not wanted:
        return build_cql(block, spaces=[], text_terms=terms)

    available = client.spaces()
    if not available:
        return build_cql(block)          # cannot verify; try as configured

    keys = {s["key"].upper(): s["key"] for s in available if s.get("key")}
    valid = [keys[w.upper()] for w in wanted if w.upper() in keys]
    missing = [w for w in wanted if w.upper() not in keys]

    if missing:
        log.warning(
            "[%s] space key(s) %s do not exist on this wiki. Readable spaces: %s",
            name, ", ".join(missing),
            ", ".join(sorted(keys.values())[:25]) or "none",
        )
    if valid:
        return build_cql(block, spaces=valid, text_terms=None)

    if terms:
        log.warning(
            "[%s] none of the configured spaces exist — searching all %d readable "
            "spaces for %s instead", name, len(available), ", ".join(terms[:5]),
        )
        return build_cql(block, spaces=[], text_terms=terms)

    log.warning(
        "[%s] none of the configured spaces exist and no `text_search` terms are "
        "set — this source will return nothing. Set `spaces` to one of the keys "
        "above, or add text_search terms.", name,
    )
    return build_cql(block, spaces=valid or wanted, text_terms=None)


def fetch(cfg: Any, block: dict, state: Any = None) -> list[Document]:
    name = block.get("name", "confluence")
    client = ConfluenceClient(block.get("base_url", ""))

    if not client.configured:
        log.warning(
            "[%s] skipped — set CONFLUENCE_BASE_URL and CONFLUENCE_TOKEN in .env",
            name,
        )
        return []
    if not client.ping():
        return []

    tags = list(block.get("tags", ["confluence"]))
    max_pages = int(block.get("max_pages", 500))
    terms = [str(t) for t in block.get("text_search", []) if str(t).strip()]

    cql = _resolve_cql(client, block, name, terms)
    log.info("[%s] CQL: %s", name, cql)

    docs: list[Document] = []

    for item in client.search(cql, limit=max_pages):
        page_id = item.get("id", "")
        title = item.get("title", "") or f"page-{page_id}"
        storage = ((item.get("body") or {}).get("storage") or {}).get("value", "")
        text = storage_to_text(storage)

        space = ((item.get("space") or {}).get("key")) or ""
        version = ((item.get("version") or {}).get("number")) or 0
        labels = [
            l.get("name", "")
            for l in (((item.get("metadata") or {}).get("labels") or {}).get("results") or [])
        ]
        webui = ((item.get("_links") or {}).get("webui")) or f"/pages/{page_id}"

        if len(text) >= 200:
            docs.append(
                Document.make(
                    title=title,
                    url=f"{client.base_url}{webui}",
                    text=text,
                    kind="confluence",
                    source=name,
                    tags=tags + [f"space:{space}"] + [f"label:{l}" for l in labels if l],
                    meta={"page_id": page_id, "space": space, "version": version, "labels": labels},
                )
            )
            log.info("  + %s (v%s, %s)", title[:60], version, space)
        else:
            log.debug("  skipped short page: %s", title)

        if block.get("include_attachments"):
            docs.extend(_attachment_docs(client, page_id, title, name, tags))

    log.info("[%s] %d documents", name, len(docs))
    return docs


def _attachment_docs(
    client: ConfluenceClient, page_id: str, page_title: str, source: str, tags: list[str]
) -> list[Document]:
    out: list[Document] = []
    for att in client.attachments(page_id):
        filename = att.get("title", "")
        ext = os.path.splitext(filename)[1].lower()
        kind = EXTENSION_MAP.get(ext)
        if kind not in ("pdf", "docx", "pptx", "xlsx"):
            continue
        link = ((att.get("_links") or {}).get("download")) or ""
        if not link:
            continue
        data = client.download(link)
        if not data:
            continue
        try:
            _, text = parse_bytes(data, kind)
        except Exception as e:
            log.debug("attachment parse failed (%s): %s", filename, e)
            continue
        if len(text) < 200:
            continue
        out.append(
            Document.make(
                title=f"{page_title} — {filename}",
                url=f"{client.base_url}{link}",
                text=text,
                kind=kind,
                source=source,
                tags=tags + ["attachment"],
                meta={"page_id": page_id, "filename": filename},
            )
        )
        log.info("    + attachment %s", filename)
    return out
