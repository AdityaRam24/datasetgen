"""Command line interface.

    python -m datagen serve               web UI to manage everything
    python -m datagen doctor              check the local model and parsers
    python -m datagen build               one incremental build
    python -m datagen build --full        ignore caches, rebuild everything
    python -m datagen agent               let the agent decide what to do
    python -m datagen scrape "<keyword>"  scrape the web for one keyword
    python -m datagen ingest <path>       ingest a file or folder
    python -m datagen learn "<text>"      teach it from your own input
    python -m datagen learn -q "…" -a "…" add a Q&A pair you wrote yourself
    python -m datagen confluence          pull the configured Confluence spaces
    python -m datagen watch               run forever, updating itself
    python -m datagen export              re-export from what is already on disk
    python -m datagen status              what the dataset currently contains
    python -m datagen inspect             sample records / see what was rejected
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config
from .llm import LocalLLM
from .state import StateStore
from .util import force_utf8_output, get_logger, setup_logging, truncate

log = get_logger("cli")


def _banner(cfg: Config, llm: LocalLLM) -> None:
    ok = llm.available()
    mark = "✓" if ok else "✗"
    print(f"\n  datagen {__version__} — dataset: {cfg.name}")
    print(f"  local model  {mark} {cfg.llm.model} via {cfg.llm.provider} @ {cfg.llm.base_url}")
    if not ok and cfg.llm.provider != "none":
        print("     (not reachable — start it with `ollama serve`; running extractive-only)")
    print()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace, cfg: Config) -> int:
    from .connectors.parsers import available
    from .connectors.confluence import ConfluenceClient

    llm = LocalLLM(cfg.llm)
    print(f"\ndatagen {__version__}  —  configuration check\n" + "─" * 60)

    print("\nLOCAL LLM  (nothing is sent to a cloud provider)")
    print(f"  provider     {cfg.llm.provider}")
    print(f"  base_url     {cfg.llm.base_url}  {'[local]' if cfg.llm.is_local else '[NOT LOCAL]'}")
    print(f"  model        {cfg.llm.model}")
    print(f"  embed model  {cfg.llm.embed_model} ({'on' if cfg.llm.embed_enabled else 'off'})")
    print(f"  vision model {cfg.llm.vision_model} "
          f"({'on — images are described' if cfg.llm.vision_enabled else 'off — images skipped'})")
    if llm.available(recheck=True):
        installed = llm.installed_models()
        print(f"  status       reachable, {len(installed)} models installed")
        for m in installed[:12]:
            marker = "  <- generation" if m.startswith(cfg.llm.model.split(":")[0]) else ""
            if cfg.llm.vision_enabled and m.startswith(cfg.llm.vision_model.split(":")[0]):
                marker = "  <- images"
            print(f"                 {m}{marker}")
        # available() resolves a missing model to an installed one, so report
        # what will actually be used rather than what the file asked for.
        if cfg.llm.model not in installed and installed:
            print(f"  WARNING      {cfg.llm.model!r} is not installed. "
                  f"Run: ollama pull {cfg.llm.model}")
        probe = llm.complete("Reply with the single word: ready", temperature=0)
        print(f"  test call    {'ok — ' + truncate(probe or '', 40) if probe else 'FAILED'}")
    else:
        print("  status       NOT REACHABLE")
        print("               start it with:  ollama serve")
        print(f"               then:           ollama pull {cfg.llm.model}")
        print(f"                               ollama pull {cfg.llm.embed_model}")

    print("\nDOCUMENT PARSERS  (builtin = works, but the library is better)")
    for kind, impl in available().items():
        flag = " " if "builtin" not in impl else "!"
        print(f"  {flag} {kind:6} {impl}")
    if any("builtin" in v for v in available().values()):
        print("    install the optional parsers:  pip install -r requirements.txt")

    print("\nSOURCES")
    from .connectors import source_blocks

    blocks = source_blocks(cfg.sources)
    if not blocks:
        print("  none enabled — edit [[sources.*]] in config.toml")
    for type_, block in blocks:
        detail = ""
        if type_ in ("files", "runbooks"):
            path = cfg.resolve(block.get("path", ""))
            detail = f"{path} {'(exists)' if path.exists() else '(MISSING — create it)'}"
        elif type_ == "web":
            detail = f"{len(block.get('seeds', []))} seeds, max {block.get('max_pages')} pages"
        elif type_ == "keywords":
            engine = block.get("engine", "duckduckgo")
            detail = f"{len(block.get('terms', []))} terms via {engine}"
            if engine == "searxng":
                from .connectors.search import searxng_available

                url = block.get("searxng_url", "")
                if searxng_available(url):
                    detail += f" @ {url} (up)"
                else:
                    detail += (
                        f" @ {url} — NOT RUNNING, will fall back to DuckDuckGo. "
                        "Start it: python searxng/setup.py"
                    )
        elif type_ == "confluence":
            client = ConfluenceClient(block.get("base_url", ""))
            detail = (
                f"{client.base_url} {'reachable' if client.ping() else 'NOT reachable'}"
                if client.configured
                else "not configured (set CONFLUENCE_BASE_URL / CONFLUENCE_TOKEN in .env)"
            )
        print(f"  {type_:11} {block.get('name', ''):22} {detail}")

    print("\nSTATE")
    with StateStore(cfg.state_db) as state:
        counts = state.counts()
        print(f"  {cfg.state_db}")
        print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
        pending = state.pending(limit=5)
        if pending:
            print(f"  {len(pending)} leads queued for the next run:")
            for row in pending:
                print(f"     [{row['type']}] {truncate(row['key'], 60)}")
    print()
    return 0


def cmd_build(args: argparse.Namespace, cfg: Config) -> int:
    from .pipeline import build

    llm = LocalLLM(cfg.llm)
    _banner(cfg, llm)
    stats = build(
        cfg,
        full=args.full,
        only=args.only,
        export=not args.no_export,
        limit=args.limit,
    )
    print("\n  " + stats.summary() + "\n")
    return 1 if stats.errors and not stats.records else 0


def cmd_agent(args: argparse.Namespace, cfg: Config) -> int:
    from .agent import run_agent

    llm = LocalLLM(cfg.llm)
    _banner(cfg, llm)
    run = run_agent(cfg, objective=args.objective, max_steps=args.max_steps)
    print(f"\n  agent finished after {len(run.steps)} steps")
    if run.summary:
        print(f"  {run.summary}")
    print()
    return 0


def _generate_and_persist(cfg: Config, llm: LocalLLM, docs: list) -> int:
    """Shared tail of the one-shot commands: chunk, dedupe, generate, gate,
    persist and export. Returns the number of accepted records."""
    from .exporters import export_all
    from .generators import generate_all
    from .models import RunStats
    from .pipeline import Pipeline
    from .quality import evaluate

    with StateStore(cfg.state_db) as state:
        pipeline = Pipeline(cfg, state, llm)
        stats = RunStats()
        chunks = pipeline.deduplicate(pipeline.to_chunks(docs, stats, full=False), stats, full=False)
        if not chunks:
            print("\n  nothing new — everything gathered is already in the dataset\n")
            return 0

        records = generate_all(chunks, llm, cfg, stats)
        accepted, quarantined = evaluate(records, llm, cfg.quality)
        all_records = pipeline.persist(chunks, accepted, quarantined)
        export_all(cfg, all_records, None)
        print(f"\n  {len(accepted)} records generated, {len(quarantined)} quarantined")
        print(f"  exports written to {cfg.export_dir}\n")
        return len(accepted)


def cmd_scrape(args: argparse.Namespace, cfg: Config) -> int:
    """Scrape a single keyword end to end — the quickest way to see it work."""
    from .connectors.search import scrape_keyword

    llm = LocalLLM(cfg.llm)
    _banner(cfg, llm)
    cfg.ensure_dirs()

    kw_cfg = cfg.sources.get("keywords", {}) or {}
    docs = scrape_keyword(
        args.keyword,
        engine=kw_cfg.get("engine", "duckduckgo"),
        searxng_url=kw_cfg.get("searxng_url", ""),
        results_per_keyword=args.results,
        max_pages=args.max_pages,
        source="cli:scrape",
        tags=["cli", "keyword"],
    )
    if not docs:
        print("  no pages could be scraped (search blocked, or no usable results)")
        return 1

    print(f"\n  scraped {len(docs)} pages:")
    for d in docs:
        print(f"    - {truncate(d.title, 70)}  ({len(d.text)} chars)  {d.url}")

    if args.no_generate:
        return 0
    _generate_and_persist(cfg, llm, docs)
    return 0


def cmd_ingest(args: argparse.Namespace, cfg: Config) -> int:
    from .connectors.files import read_file, iter_files
    from .connectors.runbooks import fetch as fetch_runbooks

    llm = LocalLLM(cfg.llm)
    _banner(cfg, llm)
    cfg.ensure_dirs()

    path = Path(args.path).expanduser()
    if not path.exists():
        print(f"  no such path: {path}")
        return 1

    tags = ["cli", "runbook"] if args.runbook else ["cli"]
    if path.is_file():
        docs = [d for d in [read_file(path, "cli:ingest", tags)] if d]
    elif args.runbook:
        docs = fetch_runbooks(cfg, {"name": "cli:runbooks", "path": str(path), "tags": tags})
    else:
        docs = [d for d in (read_file(p, "cli:ingest", tags) for p in iter_files(path, args.globs)) if d]

    if not docs:
        print("  nothing readable found")
        return 1
    print(f"\n  parsed {len(docs)} documents:")
    for d in docs[:20]:
        print(f"    - {truncate(d.title, 60):62} {len(d.text):>7} chars  [{d.kind}]")
    if len(docs) > 20:
        print(f"    … and {len(docs) - 20} more")

    if args.no_generate:
        return 0
    _generate_and_persist(cfg, llm, docs)
    return 0


def cmd_learn(args: argparse.Namespace, cfg: Config) -> int:
    """Fold input you provide into the dataset and knowledge base."""
    from .exporters import export_all
    from .learn import (
        case_record, collect_input, pair_record, read_interactive, read_stdin,
    )
    from .pipeline import Pipeline

    llm = LocalLLM(cfg.llm)
    _banner(cfg, llm)
    cfg.ensure_dirs()
    tags = args.tags or []

    # --- a pair or case you authored: trusted, stored verbatim --------------
    if args.answer or args.resolution:
        if args.resolution:
            problem = args.problem or args.text or read_stdin() or read_interactive()
            if not problem.strip():
                print("  --resolution needs a problem: pass --problem or pipe it in")
                return 1
            rec = case_record(problem, args.resolution, tags=tags)
        else:
            question = args.question or args.text
            if not question:
                print("  --answer needs a --question")
                return 1
            rec = pair_record(question, args.answer, kind=args.kind_pair, tags=tags)

        with StateStore(cfg.state_db) as state:
            pipeline = Pipeline(cfg, state, llm)
            chunk = _chunk_from_record(rec)
            all_records = pipeline.persist([chunk], [rec], [])
            export_all(cfg, all_records, None)
        print(f"\n  learned 1 {rec.kind} record (human-authored, not judged)")
        print(f"  Q: {truncate(rec.instruction, 100)}")
        print(f"  A: {truncate(rec.output, 100)}\n")
        return 0

    # --- raw material: generate from it like any other source ---------------
    text = args.text or ""
    if not text and not args.file and not args.url:
        text = read_stdin() or read_interactive()

    docs = collect_input(
        cfg, text=text, files=args.file or [], urls=args.url or [],
        kind=args.kind, title=args.title or "", tags=tags,
    )
    if not docs:
        print("\n  nothing to learn from — give me text, --file, --url, or pipe input in\n")
        return 1

    print(f"\n  learning from {len(docs)} input(s):")
    for d in docs:
        print(f"    - {truncate(d.title, 60):62} {len(d.text):>7} chars  [{d.kind}]")

    _generate_and_persist(cfg, llm, docs)
    return 0


def _chunk_from_record(rec):
    """Rebuild the chunk a human-authored record refers to, so it is persisted
    alongside the record and reaches the RAG export and the Kalam KB."""
    from .models import Chunk, Document

    doc = Document.make(
        title=rec.source_title, url=rec.source_url, text=rec.context,
        kind=rec.kind, source="human", tags=rec.tags,
    )
    chunk = Chunk.make(doc, rec.context, 0)
    chunk.id = rec.chunk_id      # keep the record's foreign key valid
    return chunk


def cmd_confluence(args: argparse.Namespace, cfg: Config) -> int:
    from .connectors.confluence import ConfluenceClient, fetch

    client = ConfluenceClient()
    if not client.configured:
        print("\n  Confluence is not configured.")
        print("  Copy .env.example to .env and set CONFLUENCE_BASE_URL and CONFLUENCE_TOKEN.\n")
        return 1
    if not client.ping():
        print(f"\n  cannot reach {client.base_url} — check the URL and the token\n")
        return 1

    blocks = cfg.sources.get("confluence", [])
    blocks = blocks if isinstance(blocks, list) else [blocks]
    if args.space:
        blocks = [{"name": "cli:confluence", "spaces": args.space,
                   "max_pages": args.limit, "include_attachments": args.attachments}]

    docs = []
    for block in blocks:
        docs.extend(fetch(cfg, {**block, "enabled": True}, None))

    print(f"\n  pulled {len(docs)} Confluence documents")
    for d in docs[:20]:
        print(f"    - {truncate(d.title, 70)}")
    if args.no_generate or not docs:
        return 0
    _generate_and_persist(cfg, LocalLLM(cfg.llm), docs)
    return 0


def cmd_watch(args: argparse.Namespace, cfg: Config) -> int:
    from .agent import watch

    if args.interval:
        cfg.schedule.interval_min = args.interval
    llm = LocalLLM(cfg.llm)
    _banner(cfg, llm)
    print(f"  updating itself every {cfg.schedule.interval_min} min — Ctrl-C to stop\n")
    watch(cfg, use_agent=not args.no_agent, once=args.once)
    return 0


def cmd_export(args: argparse.Namespace, cfg: Config) -> int:
    from .exporters import export_all

    manifest = export_all(cfg)
    print("\n" + json.dumps(manifest, indent=2)[:2000] + "\n")
    return 0


def cmd_status(args: argparse.Namespace, cfg: Config) -> int:
    print(f"\n  dataset: {cfg.name}  ({cfg.description})")
    print(f"  data:    {cfg.data_dir}")
    print(f"  exports: {cfg.export_dir}\n")

    with StateStore(cfg.state_db) as state:
        counts = state.counts()
        for key, value in counts.items():
            print(f"    {key:12} {value}")

        print("\n  recent runs")
        for run in state.recent_runs(8):
            stats = {}
            try:
                stats = json.loads(run.get("stats") or "{}")
            except json.JSONDecodeError:
                pass
            mark = "✓" if run.get("ok") else "✗"
            print(f"    {mark} {run['started_at']}  {run['mode']:12} "
                  f"records={stats.get('records', '?'):<6} docs={stats.get('docs_fetched', '?')}")

        pending = state.pending(limit=15)
        if pending:
            print(f"\n  {len(pending)} leads queued for the next run")
            for row in pending:
                print(f"    [{row['type']:7}] {truncate(row['key'], 62)}")

    manifest_path = cfg.export_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"\n  last export: {manifest.get('generated_at')}")
        print(f"    records {manifest.get('records')} "
              f"(train {manifest.get('train')} / eval {manifest.get('eval')})")
        print(f"    mean quality score {manifest.get('mean_score')}")
        print(f"    by kind: {manifest.get('kinds')}")
    print()
    return 0


def cmd_serve(args: argparse.Namespace, cfg: Config) -> int:
    """Local web UI for managing the dataset."""
    from .web import serve

    llm = LocalLLM(cfg.llm)
    _banner(cfg, llm)
    serve(cfg, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_analyze(args: argparse.Namespace, cfg: Config) -> int:
    """Is this dataset actually trainable? Report before you burn GPU hours."""
    from .analyze import analyze, dataset_card, load_records, report_text

    records = load_records(cfg.records_path)
    if not records:
        print(f"\n  no records at {cfg.records_path} — run `datagen build` first\n")
        return 1

    quarantined = len(load_records(cfg.data_dir / "quarantine.jsonl"))
    report = analyze(records, cfg, quarantined=quarantined, max_seq_len=args.max_seq_len)
    print(report_text(report))

    if args.json:
        path = cfg.export_dir / "analysis.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8")
        print(f"  wrote {path}")

    if not args.no_card:
        card = cfg.export_dir / "DATASET_CARD.md"
        card.parent.mkdir(parents=True, exist_ok=True)
        card.write_text(dataset_card(report, cfg), encoding="utf-8")
        print(f"  wrote {card}\n")

    # Non-zero when something would actually break a training run.
    return 1 if report.warnings and args.strict else 0


def cmd_inspect(args: argparse.Namespace, cfg: Config) -> int:
    path = cfg.data_dir / ("quarantine.jsonl" if args.rejected else "records.jsonl")
    if not path.exists():
        print(f"\n  {path} does not exist — run `build` first\n")
        return 1

    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if args.kind:
        rows = [r for r in rows if r.get("kind") == args.kind]

    print(f"\n  {len(rows)} records in {path.name}"
          f"{' (kind=' + args.kind + ')' if args.kind else ''}\n" + "─" * 70)

    import random as _random

    sample = _random.Random(0).sample(rows, min(args.n, len(rows))) if rows else []
    for r in sample:
        print(f"\n  [{r.get('kind')}] score={r.get('score')}  {r.get('score_reason', '')}")
        print(f"  source: {truncate(r.get('source_title', ''), 60)}  {r.get('source_url', '')}")
        print(f"  Q: {truncate(r.get('instruction', ''), 300)}")
        print(f"  A: {truncate(r.get('output', ''), 600)}")
    print()
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="datagen",
        description="Agentic dataset generator — local LLM, no cloud calls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-c", "--config", help="path to config.toml")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--log-file", help="also write logs here")
    p.add_argument("--version", action="version", version=f"datagen {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check the local model, parsers and sources").set_defaults(fn=cmd_doctor)

    b = sub.add_parser("build", help="run the pipeline over all configured sources")
    b.add_argument("--full", action="store_true", help="ignore caches, rebuild everything")
    b.add_argument("--only", nargs="+", help="limit to these source types or names")
    b.add_argument("--limit", type=int, default=0, help="cap the number of chunks (for a quick test)")
    b.add_argument("--no-export", action="store_true")
    b.set_defaults(fn=cmd_build)

    a = sub.add_parser("agent", help="let the agent plan and gather on its own")
    a.add_argument("objective", nargs="?", help="override the objective from config.toml")
    a.add_argument("--max-steps", type=int, help="step budget")
    a.set_defaults(fn=cmd_agent)

    s = sub.add_parser("scrape", help="search the web for a keyword and ingest the results")
    s.add_argument("keyword")
    s.add_argument("--results", type=int, default=10, help="search results to consider")
    s.add_argument("--max-pages", type=int, default=6, help="pages to actually download")
    s.add_argument("--no-generate", action="store_true", help="fetch only, do not build records")
    s.set_defaults(fn=cmd_scrape)

    i = sub.add_parser("ingest", help="ingest a local file or folder")
    i.add_argument("path")
    i.add_argument("--runbook", action="store_true", help="parse as runbooks (steps, rollback…)")
    i.add_argument("--globs", nargs="+", help="glob patterns when path is a folder")
    i.add_argument("--no-generate", action="store_true")
    i.set_defaults(fn=cmd_ingest)

    lr = sub.add_parser(
        "learn",
        help="teach it from your own input: text, files, URLs or a Q&A pair",
        description="Fold input you provide into the dataset and knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  datagen learn "MLIS endpoints hold their GPU until scaled to zero."
  datagen learn --file incident-2026-07.md --kind runbook
  kubectl logs pod-x | datagen learn --title "MLIS crash log" --kind log
  datagen learn --url https://docs.example.com/page
  datagen learn --question "How do I free a GPU?" --answer "Scale to zero replicas."
  datagen learn --problem "endpoint 503 after upgrade" --resolution "bucket creds expired; recreate the secret"
  datagen learn                      # interactive paste
""",
    )
    lr.add_argument("text", nargs="?", help="the text to learn from")
    lr.add_argument("--file", "-f", nargs="+", help="file(s) or folder(s) to learn from")
    lr.add_argument("--url", "-u", nargs="+", help="URL(s) to fetch and learn from")
    lr.add_argument("--title", help="title for the input")
    lr.add_argument("--kind", default="note",
                    help="note | runbook | log | doc (runbook parses steps structurally)")
    lr.add_argument("--tags", nargs="+", help="tags to attach to every record")
    lr.add_argument("--question", "-q", help="a question you are answering yourself")
    lr.add_argument("--answer", "-a", help="the answer — stored verbatim, not judged")
    lr.add_argument("--kind-pair", default="qa",
                    help="kind for a --question/--answer pair (default: qa)")
    lr.add_argument("--problem", help="a problem you hit (use with --resolution)")
    lr.add_argument("--resolution", help="how you fixed it — stored as a troubleshooting case")
    lr.set_defaults(fn=cmd_learn)

    c = sub.add_parser("confluence", help="pull Confluence spaces")
    c.add_argument("--space", nargs="+", help="space keys (defaults to config.toml)")
    c.add_argument("--limit", type=int, default=100)
    c.add_argument("--attachments", action="store_true", help="also parse PDF/DOCX attachments")
    c.add_argument("--no-generate", action="store_true")
    c.set_defaults(fn=cmd_confluence)

    w = sub.add_parser("watch", help="run forever, keeping the dataset up to date")
    w.add_argument("--interval", type=int, help="minutes between cycles")
    w.add_argument("--once", action="store_true", help="run a single cycle and exit")
    w.add_argument("--no-agent", action="store_true", help="use the plain pipeline, not the agent")
    w.set_defaults(fn=cmd_watch)

    sub.add_parser("export", help="re-export from data/ without regenerating").set_defaults(fn=cmd_export)
    sub.add_parser("status", help="show what the dataset contains").set_defaults(fn=cmd_status)

    sv = sub.add_parser("serve", help="open the web UI to manage everything")
    sv.add_argument("--port", type=int, default=8800)
    sv.add_argument("--host", default="127.0.0.1",
                    help="bind address — leave as localhost unless you know why not")
    sv.add_argument("--no-browser", action="store_true", help="do not open a browser")
    sv.set_defaults(fn=cmd_serve)

    an = sub.add_parser(
        "analyze",
        help="token stats, truncation and leakage checks, and a dataset card",
    )
    an.add_argument("--max-seq-len", type=int, default=2048,
                    help="sequence length you plan to train at (default 2048)")
    an.add_argument("--json", action="store_true", help="also write exports/analysis.json")
    an.add_argument("--no-card", action="store_true", help="skip DATASET_CARD.md")
    an.add_argument("--strict", action="store_true",
                    help="exit non-zero if any warning fires (for CI)")
    an.set_defaults(fn=cmd_analyze)

    n = sub.add_parser("inspect", help="print sample records")
    n.add_argument("-n", type=int, default=5, help="how many to show")
    n.add_argument("--kind", help="filter by kind (qa, instruction, troubleshooting)")
    n.add_argument("--rejected", action="store_true", help="show quarantined records instead")
    n.set_defaults(fn=cmd_inspect)

    return p


def main(argv: list[str] | None = None) -> int:
    # Before parse_args: `--help` prints and exits during parsing, and on a
    # cp1252 Windows console the em-dashes in the help text would crash it.
    force_utf8_output()
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose, args.log_file)
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"failed to load config: {e}", file=sys.stderr)
        return 2

    try:
        return args.fn(args, cfg)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        log.error("%s failed: %s", args.command, e, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
