"""Tools the agent can call.

Each tool is a plain Python function with a JSON-serialisable result and a
schema the planner sees. Tools are deliberately coarse-grained: a small local
model plans much better with 10 meaningful verbs than with 40 fiddly ones.

Every tool returns {"ok": bool, ...} and NEVER raises — a tool failure is an
observation the agent reasons about, not a crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from ..chunking import chunk_document
from ..config import Config
from ..connectors import files as files_conn
from ..connectors import runbooks as runbooks_conn
from ..connectors.confluence import ConfluenceClient, build_cql, storage_to_text
from ..connectors.search import scrape_keyword, web_search
from ..connectors.web import crawl, fetch_url
from ..exporters import export_all
from ..generators import generate_all
from ..llm import LocalLLM
from ..models import Chunk, Document, RunStats
from ..quality import evaluate
from ..state import StateStore
from ..util import get_logger, truncate

log = get_logger("agent.tools")


@dataclass
class ToolContext:
    cfg: Config
    state: StateStore
    llm: LocalLLM
    stats: RunStats
    # Working set, shared across tool calls within one agent run.
    documents: list[Document]
    chunks: list[Chunk]
    records: list[Any]

    def add_documents(self, docs: list[Document], label: str) -> dict:
        new, unchanged = 0, 0
        for doc in docs:
            if self.state.upsert_document(doc):
                self.documents.append(doc)
                new += 1
            else:
                unchanged += 1
        self.stats.docs_fetched += len(docs)
        self.stats.docs_skipped_unchanged += unchanged
        return {
            "ok": True,
            "fetched": len(docs),
            "new_or_changed": new,
            "unchanged": unchanged,
            "titles": [d.title[:70] for d in docs[:8]],
            "note": f"{label}: {new} documents are new or changed and are queued for chunking",
        }


ToolFn = Callable[[ToolContext, dict], dict]
REGISTRY: dict[str, tuple[ToolFn, str, dict]] = {}


def tool(name: str, description: str, params: dict) -> Callable[[ToolFn], ToolFn]:
    def wrap(fn: ToolFn) -> ToolFn:
        REGISTRY[name] = (fn, description, params)
        return fn

    return wrap


def schema_text() -> str:
    """Compact tool listing for the planner prompt."""
    lines = []
    for name, (_, desc, params) in REGISTRY.items():
        args = ", ".join(f"{k}: {v}" for k, v in params.items()) or "no arguments"
        lines.append(f"- {name}({args})\n    {desc}")
    return "\n".join(lines)


def call(ctx: ToolContext, name: str, args: dict) -> dict:
    entry = REGISTRY.get(name)
    if not entry:
        return {"ok": False, "error": f"unknown tool {name!r}; available: {list(REGISTRY)}"}
    fn = entry[0]
    try:
        result = fn(ctx, args or {})
        return result if isinstance(result, dict) else {"ok": True, "result": result}
    except Exception as e:
        log.warning("tool %s failed: %s", name, e)
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@tool(
    "search_web",
    "Search the web for a keyword and return ranked candidate URLs WITHOUT "
    "downloading them. Use this to see what exists before spending time scraping.",
    {"query": "string", "limit": "int (default 8)"},
)
def _search_web(ctx: ToolContext, args: dict) -> dict:
    query = str(args.get("query", "")).strip()
    if not query:
        return {"ok": False, "error": "query is required"}

    kw_cfg = ctx.cfg.sources.get("keywords", {}) or {}
    results = web_search(
        query,
        engine=kw_cfg.get("engine", "duckduckgo"),
        searxng_url=kw_cfg.get("searxng_url", ""),
        limit=int(args.get("limit", 8)),
    )
    return {
        "ok": True,
        "query": query,
        "count": len(results),
        "results": [
            {"url": r.url, "title": truncate(r.title, 100), "score": r.score,
             "snippet": truncate(r.snippet, 160)}
            for r in results
        ],
        "note": "nothing was downloaded; call scrape_keyword or scrape_url next"
        if results
        else "no results — the search engine may be blocked or offline",
    }


@tool(
    "scrape_keyword",
    "Search for a keyword AND download the best-ranked pages into the dataset. "
    "This is the main way to gather material on a new topic.",
    {"keyword": "string", "max_pages": "int (default 6)"},
)
def _scrape_keyword(ctx: ToolContext, args: dict) -> dict:
    keyword = str(args.get("keyword", "")).strip()
    if not keyword:
        return {"ok": False, "error": "keyword is required"}

    kw_cfg = ctx.cfg.sources.get("keywords", {}) or {}
    docs = scrape_keyword(
        keyword,
        engine=kw_cfg.get("engine", "duckduckgo"),
        searxng_url=kw_cfg.get("searxng_url", ""),
        results_per_keyword=int(args.get("max_pages", 6)) + 4,
        max_pages=int(args.get("max_pages", 6)),
        source="agent:keyword",
        tags=["agent", "keyword"],
    )
    ctx.state.propose(keyword, "keyword", "used by agent")
    ctx.state.mark(keyword, "done")
    return ctx.add_documents(docs, f"keyword {keyword!r}")


@tool(
    "scrape_url",
    "Download and parse one specific URL (HTML or PDF) into the dataset.",
    {"url": "string"},
)
def _scrape_url(ctx: ToolContext, args: dict) -> dict:
    url = str(args.get("url", "")).strip()
    if not url.startswith("http"):
        return {"ok": False, "error": "url must start with http:// or https://"}
    doc = fetch_url(url, source="agent:url", tags=["agent"])
    if not doc:
        return {"ok": False, "error": f"nothing usable at {url} (blocked, empty, or unsupported type)"}
    return ctx.add_documents([doc], "url")


@tool(
    "crawl_site",
    "Breadth-first crawl from a starting URL, staying on its domain. Use for "
    "documentation sites where one page links to the whole manual.",
    {"url": "string", "max_pages": "int (default 15)", "max_depth": "int (default 2)"},
)
def _crawl_site(ctx: ToolContext, args: dict) -> dict:
    url = str(args.get("url", "")).strip()
    if not url.startswith("http"):
        return {"ok": False, "error": "url must start with http:// or https://"}
    from urllib.parse import urlparse

    docs = crawl(
        [url],
        allow_domains=[urlparse(url).netloc],
        max_pages=int(args.get("max_pages", 15)),
        max_depth=int(args.get("max_depth", 2)),
        source="agent:crawl",
        tags=["agent", "crawl"],
        skip_urls=ctx.state.known_urls(),
    )
    return ctx.add_documents(docs, "crawl")


# ---------------------------------------------------------------------------
# Internal sources
# ---------------------------------------------------------------------------


@tool(
    "ingest_files",
    "Parse local documents (PDF, DOCX, PPTX, XLSX, MD, TXT) from a path under "
    "the project. Omit `path` to use every configured file source.",
    {"path": "string (optional)", "globs": "list of glob strings (optional)"},
)
def _ingest_files(ctx: ToolContext, args: dict) -> dict:
    blocks = []
    if args.get("path"):
        blocks.append({"name": "agent:files", "path": str(args["path"]),
                       "globs": args.get("globs"), "tags": ["agent"]})
    else:
        raw = ctx.cfg.sources.get("files", [])
        blocks = raw if isinstance(raw, list) else [raw]

    docs: list[Document] = []
    for block in blocks:
        docs.extend(files_conn.fetch(ctx.cfg, block, ctx.state))
    return ctx.add_documents(docs, "files")


@tool(
    "ingest_runbooks",
    "Parse operational runbooks, preserving symptoms, ordered steps, commands, "
    "verification and rollback. Omit `path` to use the configured runbook source.",
    {"path": "string (optional)"},
)
def _ingest_runbooks(ctx: ToolContext, args: dict) -> dict:
    blocks = []
    if args.get("path"):
        blocks.append({"name": "agent:runbooks", "path": str(args["path"]), "tags": ["runbook"]})
    else:
        raw = ctx.cfg.sources.get("runbooks", [])
        blocks = raw if isinstance(raw, list) else [raw]

    docs: list[Document] = []
    for block in blocks:
        docs.extend(runbooks_conn.fetch(ctx.cfg, block, ctx.state))
    return ctx.add_documents(docs, "runbooks")


@tool(
    "search_confluence",
    "Search the company Confluence wiki with CQL or a free-text query and ingest "
    "the matching pages (plus their PDF/DOCX attachments).",
    {"query": "string (free text)", "cql": "string (optional, raw CQL)",
     "spaces": "list of space keys (optional)", "limit": "int (default 25)"},
)
def _search_confluence(ctx: ToolContext, args: dict) -> dict:
    client = ConfluenceClient()
    if not client.configured:
        return {
            "ok": False,
            "error": "Confluence is not configured. Set CONFLUENCE_BASE_URL and "
                     "CONFLUENCE_TOKEN in dataset-generation/.env",
        }

    if args.get("cql"):
        cql = str(args["cql"])
    elif args.get("query"):
        text = str(args["query"]).replace('"', "'")
        cql = f'type = page AND text ~ "{text}"'
        if args.get("spaces"):
            joined = ", ".join(f'"{s}"' for s in args["spaces"])
            cql += f" AND space in ({joined})"
        cql += " ORDER BY lastmodified DESC"
    else:
        cql = build_cql({"spaces": args.get("spaces", [])})

    docs: list[Document] = []
    for item in client.search(cql, limit=int(args.get("limit", 25))):
        text = storage_to_text(((item.get("body") or {}).get("storage") or {}).get("value", ""))
        if len(text) < 200:
            continue
        webui = ((item.get("_links") or {}).get("webui")) or ""
        docs.append(
            Document.make(
                title=item.get("title", "untitled"),
                url=f"{client.base_url}{webui}",
                text=text,
                kind="confluence",
                source="agent:confluence",
                tags=["agent", "confluence"],
                meta={"page_id": item.get("id", ""), "cql": cql},
            )
        )
    result = ctx.add_documents(docs, "confluence")
    result["cql"] = cql
    return result


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------


@tool(
    "build_dataset",
    "Chunk everything gathered so far, remove duplicates, generate training "
    "records with the local model, and run the quality gate. Call this once you "
    "have collected enough material.",
    {},
)
def _build_dataset(ctx: ToolContext, args: dict) -> dict:
    if not ctx.documents:
        return {"ok": False, "error": "no documents gathered yet — collect sources first"}

    from ..dedupe import Deduper
    from ..pipeline import Pipeline

    pipeline = Pipeline(ctx.cfg, ctx.state, ctx.llm)
    chunks: list[Chunk] = []
    for doc in ctx.documents:
        chunks.extend(chunk_document(doc, ctx.cfg.chunking))

    chunks = pipeline.deduplicate(chunks, ctx.stats, full=False)
    if not chunks:
        return {"ok": True, "chunks": 0, "records": 0,
                "note": "everything gathered was a duplicate of existing material"}

    records = generate_all(chunks, ctx.llm, ctx.cfg, ctx.stats)
    records = Deduper().dedupe_records(records)
    accepted, quarantined = evaluate(records, ctx.llm, ctx.cfg.quality)

    # Persist immediately: if the agent runs out of steps or the planner wanders
    # off, the work is already safe on disk.
    pipeline.persist(chunks, accepted, quarantined)

    ctx.chunks.extend(chunks)
    ctx.records.extend(accepted)
    ctx.stats.records += len(accepted)
    ctx.stats.records_quarantined += len(quarantined)
    ctx.documents.clear()

    return {
        "ok": True,
        "chunks": len(chunks),
        "records": len(accepted),
        "quarantined": len(quarantined),
        "mean_score": round(sum(r.score or 0 for r in accepted) / max(1, len(accepted)), 3),
        "note": "written to disk; call export_dataset when you are done gathering",
    }


@tool(
    "assess_coverage",
    "Report what the dataset already covers and where it is thin. Use this to "
    "decide what to gather next instead of guessing.",
    {},
)
def _assess_coverage(ctx: ToolContext, args: dict) -> dict:
    counts = ctx.state.counts()
    by_kind: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for rec in ctx.records:
        by_kind[rec.kind] = by_kind.get(rec.kind, 0) + 1
        for t in rec.tags:
            if t.startswith("source:"):
                by_source[t] = by_source.get(t, 0) + 1

    topics: dict[str, int] = {}
    for chunk in ctx.chunks[-400:]:
        for token in set(chunk.tokens):
            if len(token) > 4:
                topics[token] = topics.get(token, 0) + 1
    top = sorted(topics.items(), key=lambda kv: -kv[1])[:25]

    return {
        "ok": True,
        "totals": counts,
        "records_this_run": len(ctx.records),
        "by_kind": by_kind,
        "by_source": by_source,
        "frequent_topics": [t for t, _ in top],
        "thin_kinds": [
            k for k in ctx.cfg.generation.kinds if by_kind.get(k, 0) < max(3, len(ctx.records) // 10)
        ],
        "pending_leads": len(ctx.state.pending()),
    }


@tool(
    "propose_sources",
    "Record keywords or URLs worth exploring on the NEXT run. This is how the "
    "generator improves itself over time — use it for gaps you cannot fill now.",
    {"keywords": "list of strings", "urls": "list of strings", "reason": "string"},
)
def _propose_sources(ctx: ToolContext, args: dict) -> dict:
    reason = str(args.get("reason", "proposed by agent"))[:200]
    added = {"keywords": [], "urls": []}
    limit = ctx.cfg.agent.max_proposed_per_run

    for kw in (args.get("keywords") or [])[:limit]:
        kw = str(kw).strip()
        if kw and ctx.state.propose(kw, "keyword", reason, score=0.5):
            added["keywords"].append(kw)
    for url in (args.get("urls") or [])[:limit]:
        url = str(url).strip()
        if url.startswith("http") and ctx.state.propose(url, "url", reason, score=0.5):
            added["urls"].append(url)

    return {
        "ok": True,
        "added": added,
        "note": f"{len(added['keywords']) + len(added['urls'])} new leads stored; "
                "duplicates of known leads were ignored",
    }


@tool(
    "export_dataset",
    "Write the dataset to disk in every configured format and update the Kalam "
    "PCAI knowledge base. Call this last.",
    {},
)
def _export_dataset(ctx: ToolContext, args: dict) -> dict:
    # None/None: export everything on disk, not just what this run produced.
    manifest = export_all(ctx.cfg)
    return {
        "ok": True,
        "records": manifest.get("records"),
        "chunks": manifest.get("chunks"),
        "out_dir": str(ctx.cfg.export_dir),
    }


@tool(
    "finish",
    "End the run. Call this when the objective is met or no useful work remains.",
    {"summary": "string — what was accomplished"},
)
def _finish(ctx: ToolContext, args: dict) -> dict:
    return {"ok": True, "done": True, "summary": str(args.get("summary", "run complete"))[:500]}


def tools_json() -> str:
    return json.dumps(
        [{"name": n, "description": d, "parameters": p} for n, (_, d, p) in REGISTRY.items()],
        indent=2,
    )
