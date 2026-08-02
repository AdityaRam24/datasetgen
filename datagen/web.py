"""Local web UI — `python -m datagen serve`.

A small control panel for the things you otherwise do on the command line:
upload documents, manage keywords, kick off a build, watch the log, browse the
records that came out, and check whether the result is trainable.

Design constraints, same as the rest of the project:

  * stdlib only — http.server, no Flask/FastAPI dependency
  * bound to 127.0.0.1 by default. This UI writes files and starts jobs, so it
    is deliberately not reachable from the network
  * no shell execution anywhere: buttons call Python functions directly, so
    there is no command string for anything to be injected into
  * uploads are extension-allowlisted, size-capped, and path-checked against
    the corpus directory before anything is written

Long jobs run on a background thread with their log captured, so the page can
poll rather than hold a request open.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

import logging

from .config import Config, load_config
from .llm import LocalLLM
from .state import StateStore
from .util import get_logger, human_bytes, now_iso, truncate

log = get_logger("web")

UI_DIR = Path(__file__).resolve().parent / "webui"

# Uploads: what we will accept, and how much of it. Deliberately an allowlist —
# it must stay in step with EXTENSION_MAP, or the UI refuses files the pipeline
# would happily read. Images are here because a vision model describes them
# ([llm.vision]); dragging a screenshot in is the obvious way to hand one over.
ALLOWED_UPLOAD_EXT = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".md", ".markdown", ".txt",
    ".html", ".htm", ".csv", ".json", ".log", ".rst", ".yaml", ".yml",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
}
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_BODY_BYTES = 120 * 1024 * 1024      # base64 inflates by ~33%


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


class LogCapture(logging.Handler):
    """Mirrors the datagen logger into a ring buffer the browser can poll."""

    def __init__(self, buffer: deque) -> None:
        super().__init__()
        self.buffer = buffer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.buffer.append(
                {
                    "t": time.strftime("%H:%M:%S"),
                    "level": record.levelname,
                    "msg": record.getMessage()[:2000],
                }
            )
        except Exception:
            pass


@dataclass
class JobRunner:
    """One job at a time — the local model is a single resource, and two
    concurrent builds would fight over it and over the SQLite state."""

    lines: deque = field(default_factory=lambda: deque(maxlen=4000))
    current: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str = ""
    _thread: threading.Thread | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def install(self) -> None:
        handler = LogCapture(self.lines)
        handler.setLevel(logging.INFO)
        logging.getLogger("datagen").addHandler(handler)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def note(self, msg: str, level: str = "INFO") -> None:
        self.lines.append({"t": time.strftime("%H:%M:%S"), "level": level, "msg": msg})

    def start(self, name: str, fn: Callable[[], Any]) -> tuple[bool, str]:
        with self._lock:
            if self.running:
                return False, f"'{self.current}' is still running"

            def wrap() -> None:
                self.error = ""
                self.started_at = time.time()
                self.finished_at = 0.0
                self.note(f"=== {name} started ===")
                try:
                    fn()
                    self.note(f"=== {name} finished in {time.time() - self.started_at:.0f}s ===")
                except Exception as e:
                    self.error = f"{type(e).__name__}: {e}"
                    self.note(f"!!! {name} failed: {self.error}", "ERROR")
                    log.error("job %s failed", name, exc_info=True)
                finally:
                    self.finished_at = time.time()

            self.current = name
            self._thread = threading.Thread(target=wrap, daemon=True, name=f"job-{name}")
            self._thread.start()
            return True, name

    def snapshot(self, offset: int = 0) -> dict:
        lines = list(self.lines)
        offset = max(0, min(offset, len(lines)))
        return {
            "running": self.running,
            "current": self.current,
            "error": self.error,
            "elapsed": round((self.finished_at or time.time()) - self.started_at, 1)
            if self.started_at
            else 0,
            "lines": lines[offset:],
            "next": len(lines),
        }


# ---------------------------------------------------------------------------
# config.toml editing
# ---------------------------------------------------------------------------

_KEYWORDS_BLOCK = re.compile(
    r"(\[sources\.keywords\](?:.|\n)*?terms\s*=\s*)\[[^\]]*\]", re.M
)


def render_terms(terms: list[str]) -> str:
    if not terms:
        return "[]"
    body = ",\n".join(f'  "{t.strip()}"' for t in terms if t.strip())
    return f"[\n{body},\n]"


def update_keywords(config_path: Path, terms: list[str]) -> bool:
    """Rewrite the `terms = [...]` list in [sources.keywords].

    tomllib is read-only and pulling in a TOML *writer* to change one list is
    not worth the dependency, so this is a targeted replacement that leaves
    every comment and every other setting in the file untouched.
    """
    text = config_path.read_text(encoding="utf-8")
    if not _KEYWORDS_BLOCK.search(text):
        return False
    updated = _KEYWORDS_BLOCK.sub(lambda m: m.group(1) + render_terms(terms), text, count=1)
    if updated == text:
        return False
    config_path.write_text(updated, encoding="utf-8")
    return True


def render_toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(f'"{str(v)}"' for v in value) + "]"
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _section_bounds(text: str, section: str) -> tuple[int, int] | None:
    """Character span of a `[section]` or `[[section]]` table body."""
    header = re.compile(
        rf"^\[\[?{re.escape(section)}\]\]?\s*$", re.M
    )
    m = header.search(text)
    if not m:
        return None
    start = m.end()
    nxt = re.compile(r"^\[", re.M).search(text, start)
    return start, (nxt.start() if nxt else len(text))


# A key's value may be a single line, a bracketed list spanning lines, or a
# triple-quoted string. Matching all three keeps us from corrupting the file.
def _value_span(body: str, key: str) -> tuple[int, int, int] | None:
    m = re.search(rf"^(\s*{re.escape(key)}\s*=\s*)", body, re.M)
    if not m:
        return None
    vstart = m.end()
    rest = body[vstart:]

    if rest.lstrip().startswith('"""'):
        lead = len(rest) - len(rest.lstrip())
        close = rest.find('"""', lead + 3)
        end = vstart + (close + 3 if close != -1 else len(rest))
    elif rest.lstrip().startswith("["):
        depth, i = 0, len(rest) - len(rest.lstrip())
        while i < len(rest):
            if rest[i] == "[":
                depth += 1
            elif rest[i] == "]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        end = vstart + i
    else:
        nl = rest.find("\n")
        line = rest if nl == -1 else rest[:nl]
        # Keep any trailing comment on the line.
        hash_at = line.find("#")
        end = vstart + (len(line.rstrip()) if hash_at == -1 else hash_at)
    return m.start(), vstart, end


def update_toml(config_path: Path, section: str, changes: dict[str, Any]) -> list[str]:
    """Set keys inside one TOML table, preserving comments and everything else.

    tomllib is read-only, and a TOML *writer* would reformat the whole file and
    throw away the comments that make config.toml self-documenting. So this
    edits in place: existing keys have just their value replaced, missing keys
    are appended to the end of the table.

    Returns the list of keys actually changed. Raises if the section is missing
    or the result would not parse.
    """
    import tomllib

    text = config_path.read_text(encoding="utf-8")
    bounds = _section_bounds(text, section)
    if not bounds:
        raise ValueError(f"[{section}] not found in {config_path.name}")

    start, end = bounds
    body = text[start:end]
    changed: list[str] = []

    for key, value in changes.items():
        if value is None:
            continue
        rendered = render_toml_value(value)
        span = _value_span(body, key)
        if span:
            _, vstart, vend = span
            if body[vstart:vend].strip() == rendered:
                continue
            body = body[:vstart] + rendered + body[vend:]
        else:
            body = body.rstrip("\n") + f"\n{key} = {rendered}\n"
        changed.append(key)

    if not changed:
        return []

    updated = text[:start] + body + text[end:]
    try:
        tomllib.loads(updated)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"refusing to write invalid TOML: {e}") from e

    config_path.write_text(updated, encoding="utf-8")
    return changed


def update_env(env_path: Path, values: dict[str, str]) -> list[str]:
    """Set KEY=VALUE lines in .env, preserving comments and ordering.

    Also applies the values to the current process, because .env is only read
    at startup and the caller expects the change to take effect now.
    """
    import os

    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    changed: list[str] = []

    for key, value in values.items():
        if value is None:
            continue
        value = str(value)
        line = f"{key}={value}"
        for i, existing in enumerate(lines):
            stripped = existing.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            if stripped.split("=", 1)[0].strip() == key:
                if existing != line:
                    lines[i] = line
                    changed.append(key)
                break
        else:
            lines.append(line)
            changed.append(key)
        os.environ[key] = value          # .env is only read at startup

    if changed:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def mask_secret(value: str) -> str:
    """Never send a stored token back to the browser in full."""
    if not value:
        return ""
    if len(value) <= 8:
        return "•" * len(value)
    return f"{value[:4]}{'•' * 12}{value[-4:]}"


def update_project(config_path: Path, name: str | None, description: str | None) -> bool:
    text = original = config_path.read_text(encoding="utf-8")
    if name:
        text = re.sub(r'(?m)^(name\s*=\s*)".*"', lambda m: m.group(1) + f'"{name}"', text, count=1)
    if description is not None:
        text = re.sub(
            r'(?m)^(description\s*=\s*)".*"',
            lambda m: m.group(1) + f'"{description}"',
            text,
            count=1,
        )
    if text != original:
        config_path.write_text(text, encoding="utf-8")
        return True
    return False


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


def safe_upload_path(root: Path, folder: str, filename: str) -> Path:
    """Resolve an upload target, refusing anything outside the corpus.

    Browsers can be told to send any filename, so this strips directory
    components, rejects unknown extensions, and verifies the resolved path is
    still inside `root` before the caller writes a byte.
    """
    if folder not in ("docs", "runbooks"):
        raise ValueError("folder must be 'docs' or 'runbooks'")

    name = Path(unquote(filename)).name           # drops ../ and any directory
    name = re.sub(r'[<>:"|?*\x00-\x1f]', "_", name).strip(". ")
    if not name:
        raise ValueError("empty filename")

    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXT:
        raise ValueError(
            f"'{suffix or name}' is not an accepted type "
            f"({', '.join(sorted(ALLOWED_UPLOAD_EXT))})"
        )

    target = (root / folder / name).resolve()
    if not str(target).startswith(str(root.resolve())):
        raise ValueError("path escapes the corpus directory")
    return target


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class DatagenServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, cfg: Config) -> None:
        super().__init__(addr, handler)
        self.cfg = cfg
        self.jobs = JobRunner()
        self.jobs.install()
        self.llm = LocalLLM(cfg.llm)


class Handler(BaseHTTPRequestHandler):
    server: DatagenServer
    server_version = "datagen"

    # -- plumbing -----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # This UI is local-only; refuse to be embedded or probed cross-origin.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, data: Any, code: int = 200) -> None:
        self._send(code, json.dumps(data, default=str).encode("utf-8"), "application/json")

    def _error(self, message: str, code: int = 400) -> None:
        self._json({"ok": False, "error": message}, code)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError(f"request too large ({human_bytes(length)})")
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"invalid JSON body: {e}") from e

    @property
    def cfg(self) -> Config:
        return self.server.cfg

    @property
    def jobs(self) -> JobRunner:
        return self.server.jobs

    # -- routing ------------------------------------------------------------

    def do_GET(self) -> None:
        route = urlparse(self.path)
        path, query = route.path, parse_qs(route.query)
        try:
            if path in ("/", "/index.html"):
                return self._serve_ui()
            if path == "/api/state":
                return self._json(self._state())
            if path == "/api/logs":
                return self._json(self.jobs.snapshot(int((query.get("offset") or ["0"])[0])))
            if path == "/api/records":
                return self._json(self._records(query))
            if path == "/api/analysis":
                return self._json(self._analysis(query))
            if path == "/api/exports":
                return self._json(self._exports())
            if path == "/api/models":
                return self._json(self._models())
            if path == "/api/settings":
                return self._json(self._settings())
            if path == "/api/confluence":
                return self._json(self._confluence_get())
            if path == "/api/confluence/spaces":
                return self._json(self._confluence_spaces())
            if path.startswith("/download/"):
                return self._download(path[len("/download/"):])
            return self._error("not found", 404)
        except Exception as e:
            log.error("GET %s failed: %s", path, e, exc_info=True)
            self._error(str(e), 500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._body()
            if path == "/api/run":
                return self._json(self._run(body))
            if path == "/api/upload":
                return self._json(self._upload(body))
            if path == "/api/keywords":
                return self._json(self._keywords(body))
            if path == "/api/project":
                return self._json(self._project(body))
            if path == "/api/learn":
                return self._json(self._learn(body))
            if path == "/api/settings":
                return self._json(self._settings_save(body))
            if path == "/api/confluence":
                return self._json(self._confluence_save(body))
            if path == "/api/confluence/test":
                return self._json(self._confluence_test())
            return self._error("not found", 404)
        except ValueError as e:
            self._error(str(e), 400)
        except Exception as e:
            log.error("POST %s failed: %s", path, e, exc_info=True)
            self._error(str(e), 500)

    # -- handlers -----------------------------------------------------------

    def _serve_ui(self) -> None:
        page = UI_DIR / "index.html"
        if not page.is_file():
            return self._error("UI file missing — expected datagen/webui/index.html", 500)
        self._send(200, page.read_bytes(), "text/html; charset=utf-8")

    def _state(self) -> dict:
        cfg = self.cfg
        with StateStore(cfg.state_db) as state:
            counts = state.counts()
            runs = state.recent_runs(5)
            pending = state.pending(limit=10)

        kw = cfg.sources.get("keywords", {}) or {}
        searxng = False
        if kw.get("engine") == "searxng":
            from .connectors.search import searxng_available

            searxng = searxng_available(kw.get("searxng_url", ""))

        corpus = {}
        for folder in ("docs", "runbooks"):
            root = cfg.resolve(f"corpus/{folder}")
            files = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
            corpus[folder] = {
                "path": str(root),
                "count": len(files),
                "bytes": sum(p.stat().st_size for p in files),
                "recent": [p.name for p in sorted(files, key=lambda x: -x.stat().st_mtime)[:8]],
            }

        return {
            "ok": True,
            "project": {"name": cfg.name, "description": cfg.description},
            "llm": {
                "model": cfg.llm.model,
                "provider": cfg.llm.provider,
                "base_url": cfg.llm.base_url,
                "available": self.server.llm.available(),
            },
            "search": {"engine": kw.get("engine", "duckduckgo"), "up": searxng,
                       "url": kw.get("searxng_url", "")},
            "keywords": list(kw.get("terms", [])),
            "counts": counts,
            "corpus": corpus,
            "runs": runs,
            "pending": pending,
            "job": self.jobs.snapshot(10**9),   # status only, no log replay
        }

    def _run(self, body: dict) -> dict:
        action = str(body.get("action", "")).strip()
        cfg = self.cfg
        llm = self.server.llm

        def build(full: bool = False) -> Callable[[], Any]:
            from .pipeline import build as run_build

            return lambda: run_build(cfg, full=full, export=True)

        if action == "build":
            ok, msg = self.jobs.start("build", build(False))
        elif action == "build_full":
            ok, msg = self.jobs.start("full rebuild", build(True))
        elif action == "agent":
            from .agent import run_agent

            ok, msg = self.jobs.start("agent", lambda: run_agent(cfg))
        elif action == "export":
            from .exporters import export_all

            ok, msg = self.jobs.start("export", lambda: export_all(cfg))
        elif action == "scrape":
            keyword = str(body.get("keyword", "")).strip()
            if not keyword:
                raise ValueError("keyword is required")
            ok, msg = self.jobs.start(f"scrape {keyword!r}", self._scrape_job(keyword))
        else:
            raise ValueError(f"unknown action {action!r}")

        return {"ok": ok, "message": msg, "started": ok}

    def _scrape_job(self, keyword: str) -> Callable[[], Any]:
        cfg, llm = self.cfg, self.server.llm

        def job() -> None:
            from .connectors.search import scrape_keyword
            from .exporters import export_all
            from .generators import generate_all
            from .models import RunStats
            from .pipeline import Pipeline
            from .quality import evaluate

            kw = cfg.sources.get("keywords", {}) or {}
            docs = scrape_keyword(
                keyword,
                engine=kw.get("engine", "duckduckgo"),
                searxng_url=kw.get("searxng_url", ""),
                max_pages=int(kw.get("max_pages_per_keyword", 6)),
                source="web-ui",
                tags=["web-ui", "keyword"],
            )
            if not docs:
                log.warning("no pages scraped for %r", keyword)
                return
            with StateStore(cfg.state_db) as state:
                pipeline = Pipeline(cfg, state, llm)
                stats = RunStats()
                chunks = pipeline.deduplicate(
                    pipeline.to_chunks(docs, stats, full=False), stats, full=False
                )
                if not chunks:
                    log.info("everything scraped was already in the dataset")
                    return
                # Checkpointed, like every other generation path: a long
                # scrape started from the UI must survive the server stopping.
                all_records, _ = pipeline.generate_in_batches(chunks, stats)
                export_all(cfg, all_records, None)

        return job

    def _upload(self, body: dict) -> dict:
        folder = str(body.get("folder", "docs"))
        filename = str(body.get("name", ""))
        data_b64 = body.get("data") or ""
        if not filename or not data_b64:
            raise ValueError("name and data are required")

        root = self.cfg.resolve("corpus")
        target = safe_upload_path(root, folder, filename)

        try:
            blob = base64.b64decode(data_b64, validate=True)
        except Exception as e:
            raise ValueError(f"could not decode upload: {e}") from e
        if len(blob) > MAX_UPLOAD_BYTES:
            raise ValueError(f"file is {human_bytes(len(blob))}, limit is "
                             f"{human_bytes(MAX_UPLOAD_BYTES)}")
        if not blob:
            raise ValueError("file is empty")

        target.parent.mkdir(parents=True, exist_ok=True)
        replaced = target.exists()
        target.write_bytes(blob)
        self.jobs.note(f"uploaded {target.name} ({human_bytes(len(blob))}) to corpus/{folder}")
        log.info("upload: %s -> %s", target.name, folder)

        return {"ok": True, "path": str(target), "bytes": len(blob), "replaced": replaced}

    def _keywords(self, body: dict) -> dict:
        terms = body.get("terms")
        if not isinstance(terms, list):
            raise ValueError("terms must be a list")
        clean = [str(t).strip() for t in terms if str(t).strip()][:200]

        path = self.cfg.root / "config.toml"
        if not update_keywords(path, clean):
            raise ValueError(
                "could not locate the `terms = [...]` list under [sources.keywords] "
                "in config.toml — edit it by hand"
            )
        self.server.cfg = load_config(str(path))     # pick the change up immediately
        self.jobs.note(f"keywords updated ({len(clean)} terms)")
        return {"ok": True, "terms": clean}

    def _project(self, body: dict) -> dict:
        name = str(body.get("name", "")).strip() or None
        description = body.get("description")
        description = str(description).strip() if description is not None else None
        path = self.cfg.root / "config.toml"
        changed = update_project(path, name, description)
        self.server.cfg = load_config(str(path))
        return {"ok": True, "changed": changed,
                "project": {"name": self.server.cfg.name,
                            "description": self.server.cfg.description}}

    def _learn(self, body: dict) -> dict:
        from .exporters import export_all
        from .learn import case_record, document_from_text, pair_record
        from .pipeline import Pipeline

        cfg, llm = self.cfg, self.server.llm
        mode = str(body.get("mode", "text"))
        tags = [str(t).strip() for t in (body.get("tags") or []) if str(t).strip()]

        if mode == "pair":
            question = str(body.get("question", "")).strip()
            answer = str(body.get("answer", "")).strip()
            if not question or not answer:
                raise ValueError("both a question and an answer are required")
            rec = pair_record(question, answer, tags=tags)
        elif mode == "case":
            problem = str(body.get("problem", "")).strip()
            resolution = str(body.get("resolution", "")).strip()
            if not problem or not resolution:
                raise ValueError("both a problem and a resolution are required")
            rec = case_record(problem, resolution, tags=tags)
        else:
            text = str(body.get("text", "")).strip()
            if len(text) < 40:
                raise ValueError("that is too short to learn anything from (40+ characters)")
            doc = document_from_text(text, title=str(body.get("title", "")).strip(), tags=tags)
            if not doc:
                raise ValueError("could not build a document from that text")

            def job() -> None:
                from .generators import generate_all
                from .models import RunStats
                from .quality import evaluate

                with StateStore(cfg.state_db) as state:
                    pipeline = Pipeline(cfg, state, llm)
                    stats = RunStats()
                    chunks = pipeline.deduplicate(
                        pipeline.to_chunks([doc], stats, full=False), stats, full=False
                    )
                    if not chunks:
                        log.info("that input is already in the dataset")
                        return
                    all_records, _ = pipeline.generate_in_batches(chunks, stats)
                    export_all(cfg, all_records, None)

            ok, msg = self.jobs.start("learn", job)
            return {"ok": ok, "message": msg, "async": True}

        # Human-authored pairs are instant — no model involved.
        from .__main__ import _chunk_from_record

        with StateStore(cfg.state_db) as state:
            pipeline = Pipeline(cfg, state, llm)
            all_records = pipeline.persist([_chunk_from_record(rec)], [rec], [])
            export_all(cfg, all_records, None)
        self.jobs.note(f"learned a {rec.kind} record: {truncate(rec.instruction, 60)}")
        return {"ok": True, "kind": rec.kind, "id": rec.id, "async": False}

    # -- settings -----------------------------------------------------------

    def _reload(self) -> None:
        """Re-read config.toml and rebuild the LLM client after a settings save."""
        self.server.cfg = load_config(str(self.cfg.root / "config.toml"))
        self.server.llm = LocalLLM(self.server.cfg.llm)

    def _models(self) -> dict:
        llm = self.server.llm
        models = llm.model_catalog()
        return {
            "ok": True,
            "reachable": llm.available(recheck=True),
            "base_url": llm.cfg.base_url,
            "provider": llm.cfg.provider,
            "models": models,
            "chat": [m for m in models if not m["embedding"]],
            "embedding": [m for m in models if m["embedding"]],
        }

    def _settings(self) -> dict:
        cfg = self.cfg
        return {
            "ok": True,
            "llm": {
                "provider": cfg.llm.provider,
                "base_url": cfg.llm.base_url,
                "model": cfg.llm.model,
                "embed_model": cfg.llm.embed_model,
                "embed_enabled": cfg.llm.embed_enabled,
                "temperature": cfg.llm.temperature,
                "num_ctx": cfg.llm.num_ctx,
                "available": self.server.llm.available(),
            },
            "generation": {
                "kinds": cfg.generation.kinds,
                "pairs_per_chunk": cfg.generation.pairs_per_chunk,
                "max_workers": cfg.generation.max_workers,
            },
            "chunking": {"max_chars": cfg.chunking.max_chars, "overlap": cfg.chunking.overlap},
            "quality": {"enabled": cfg.quality.enabled, "min_score": cfg.quality.min_score},
            "project": {"name": cfg.name, "description": cfg.description},
            "config_path": str(cfg.root / "config.toml"),
        }

    def _settings_save(self, body: dict) -> dict:
        path = self.cfg.root / "config.toml"
        changed: list[str] = []

        llm = body.get("llm") or {}
        if llm:
            provider = str(llm.get("provider") or "").strip() or None
            base_url = str(llm.get("base_url") or "").strip() or None
            if provider and provider not in ("ollama", "openai_compat", "none"):
                raise ValueError("provider must be ollama, openai_compat or none")
            if base_url:
                # This project is local-only by design; refuse a cloud endpoint
                # rather than silently sending the corpus somewhere.
                from .config import LLMConfig

                if not LLMConfig(base_url=base_url).is_local:
                    raise ValueError(
                        f"{base_url} is not a local or private address. This tool is "
                        "local-LLM only — use localhost, a LAN address, or a .local host."
                    )
            changed += update_toml(path, "llm", {
                "provider": provider,
                "base_url": base_url,
                "model": str(llm.get("model") or "").strip() or None,
                "temperature": llm.get("temperature"),
                "num_ctx": llm.get("num_ctx"),
            })
            embed = {}
            if llm.get("embed_model") is not None:
                embed["model"] = str(llm["embed_model"]).strip()
            if llm.get("embed_enabled") is not None:
                embed["enabled"] = bool(llm["embed_enabled"])
            if embed:
                changed += update_toml(path, "llm.embeddings", embed)

        gen = body.get("generation") or {}
        if gen:
            changed += update_toml(path, "generation", {
                "kinds": gen.get("kinds"),
                "pairs_per_chunk": gen.get("pairs_per_chunk"),
                "max_workers": gen.get("max_workers"),
            })

        chunking = body.get("chunking") or {}
        if chunking:
            changed += update_toml(path, "chunking", {
                "max_chars": chunking.get("max_chars"),
                "overlap": chunking.get("overlap"),
            })

        quality = body.get("quality") or {}
        if quality:
            changed += update_toml(path, "quality", {
                "enabled": quality.get("enabled"),
                "min_score": quality.get("min_score"),
            })

        project = body.get("project") or {}
        if project:
            if update_project(path, str(project.get("name") or "").strip() or None,
                              project.get("description")):
                changed.append("project")

        self._reload()
        if changed:
            self.jobs.note(f"settings saved: {', '.join(changed)}")
        return {"ok": True, "changed": changed, "settings": self._settings()}

    # -- confluence ---------------------------------------------------------

    def _confluence_block(self) -> dict:
        blocks = self.cfg.sources.get("confluence") or []
        blocks = blocks if isinstance(blocks, list) else [blocks]
        return blocks[0] if blocks else {}

    def _confluence_get(self) -> dict:
        import os

        from .connectors.confluence import ConfluenceClient, build_cql

        block = self._confluence_block()
        client = ConfluenceClient()
        token = os.getenv("CONFLUENCE_TOKEN", "")
        return {
            "ok": True,
            "configured": client.configured,
            "credentials": {
                "base_url": client.base_url,
                "user": client.user,
                "token_set": bool(token),
                "token_masked": mask_secret(token),
                "mode": "cloud (basic auth)" if client.user else
                        ("server/DC (bearer token)" if token else "not set"),
            },
            "source": {
                "enabled": bool(block.get("enabled", False)),
                "spaces": list(block.get("spaces", [])),
                "cql": block.get("cql", ""),
                "max_pages": int(block.get("max_pages", 25)),
                "include_attachments": bool(block.get("include_attachments", True)),
                "updated_since": block.get("updated_since", ""),
            },
            "effective_cql": build_cql(block),
            "env_path": str(self.cfg.root / ".env"),
        }

    def _confluence_save(self, body: dict) -> dict:
        creds = body.get("credentials") or {}
        env_changes: dict[str, str] = {}
        for field, key in (("base_url", "CONFLUENCE_BASE_URL"),
                           ("user", "CONFLUENCE_USER"),
                           ("token", "CONFLUENCE_TOKEN")):
            if field in creds and creds[field] is not None:
                value = str(creds[field]).strip()
                # A masked token means "unchanged" — never write the dots back.
                if field == "token" and ("•" in value or not value):
                    continue
                if field == "base_url":
                    value = value.rstrip("/")
                    if value and not value.startswith(("http://", "https://")):
                        raise ValueError("base URL must start with http:// or https://")
                env_changes[key] = value

        changed_env = update_env(self.cfg.root / ".env", env_changes) if env_changes else []

        source = body.get("source") or {}
        changed_cfg: list[str] = []
        if source:
            spaces = source.get("spaces")
            if isinstance(spaces, str):
                spaces = [s.strip() for s in spaces.split(",") if s.strip()]
            changed_cfg = update_toml(
                self.cfg.root / "config.toml", "sources.confluence",
                {
                    "enabled": source.get("enabled"),
                    "spaces": spaces,
                    "cql": source.get("cql"),
                    "max_pages": source.get("max_pages"),
                    "include_attachments": source.get("include_attachments"),
                    "updated_since": source.get("updated_since"),
                },
            )

        self._reload()
        note = ", ".join(changed_env + changed_cfg)
        if note:
            self.jobs.note(f"confluence settings saved: {note}")
        return {"ok": True, "changed": changed_env + changed_cfg,
                "confluence": self._confluence_get()}

    def _confluence_test(self) -> dict:
        from .connectors.confluence import ConfluenceClient

        client = ConfluenceClient()
        if not client.configured:
            return {"ok": False, "error":
                    "Set the base URL and token first — then Test again."}
        if not client.ping():
            return {"ok": False, "error":
                    f"Could not reach {client.base_url}. Check the URL, the token, and "
                    "(for Cloud) that the email address is right. Server/Data Center "
                    "needs the user field EMPTY."}
        spaces = self._confluence_spaces()
        return {
            "ok": True,
            "base_url": client.base_url,
            "mode": "cloud" if client.user else "server/DC",
            "spaces_visible": len(spaces.get("spaces", [])),
            "message": f"Connected to {client.base_url} — "
                       f"{len(spaces.get('spaces', []))} spaces visible.",
        }

    def _confluence_spaces(self) -> dict:
        """List spaces the token can see, so keys can be picked rather than typed."""
        from .connectors.confluence import ConfluenceClient
        from .util import HttpError, http_request

        client = ConfluenceClient()
        if not client.configured:
            return {"ok": False, "error": "not configured", "spaces": []}
        try:
            data = http_request(
                f"{client.base_url}/rest/api/space?limit=200",
                headers=client.headers, timeout=30, retries=0,
            ).json()
        except (HttpError, ValueError) as e:
            return {"ok": False, "error": str(e), "spaces": []}

        return {
            "ok": True,
            "spaces": [
                {"key": s.get("key", ""), "name": s.get("name", ""), "type": s.get("type", "")}
                for s in data.get("results", []) if s.get("key")
            ],
        }

    def _records(self, query: dict) -> dict:
        from .analyze import load_records

        rejected = (query.get("rejected") or ["0"])[0] == "1"
        path = self.cfg.data_dir / ("quarantine.jsonl" if rejected else "records.jsonl")
        records = load_records(path)

        kind = (query.get("kind") or [""])[0]
        search = (query.get("q") or [""])[0].lower()
        limit = min(int((query.get("limit") or ["50"])[0]), 500)

        if kind:
            records = [r for r in records if r.kind == kind]
        if search:
            records = [
                r for r in records
                if search in r.instruction.lower() or search in r.output.lower()
            ]

        kinds: dict[str, int] = {}
        for r in load_records(self.cfg.data_dir / "records.jsonl"):
            kinds[r.kind] = kinds.get(r.kind, 0) + 1

        return {
            "ok": True,
            "total": len(records),
            "kinds": kinds,
            "records": [
                {
                    "id": r.id, "kind": r.kind, "instruction": r.instruction,
                    "output": r.output, "score": r.score, "reason": r.score_reason,
                    "source_title": r.source_title, "source_url": r.source_url,
                    "generator": r.generator, "tags": r.tags,
                }
                for r in records[-limit:][::-1]
            ],
        }

    def _analysis(self, query: dict) -> dict:
        from .analyze import analyze, load_records

        records = load_records(self.cfg.records_path)
        if not records:
            return {"ok": False, "error": "no records yet — run a build first"}
        quarantined = len(load_records(self.cfg.data_dir / "quarantine.jsonl"))
        max_seq = int((query.get("max_seq_len") or ["2048"])[0])
        report = analyze(records, self.cfg, quarantined=quarantined, max_seq_len=max_seq)
        return {"ok": True, "report": report.to_dict()}

    def _exports(self) -> dict:
        out_dir = self.cfg.export_dir
        files = []
        if out_dir.exists():
            for p in sorted(out_dir.iterdir()):
                if p.is_file():
                    files.append({
                        "name": p.name,
                        "bytes": p.stat().st_size,
                        "human": human_bytes(p.stat().st_size),
                        "modified": time.strftime("%Y-%m-%d %H:%M",
                                                  time.localtime(p.stat().st_mtime)),
                    })
        return {"ok": True, "dir": str(out_dir), "files": files}

    def _download(self, name: str) -> None:
        out_dir = self.cfg.export_dir.resolve()
        target = (out_dir / Path(unquote(name)).name).resolve()
        if not str(target).startswith(str(out_dir)) or not target.is_file():
            return self._error("not found", 404)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(
            200, target.read_bytes(), ctype,
            {"Content-Disposition": f'attachment; filename="{target.name}"'},
        )


def serve(cfg: Config, host: str = "127.0.0.1", port: int = 8800, open_browser: bool = True) -> None:
    httpd = DatagenServer((host, port), Handler, cfg)
    url = f"http://{host}:{port}"

    print(f"\n  Dataset manager running at {url}")
    print(f"  dataset: {cfg.name}")
    print(f"  corpus:  {cfg.resolve('corpus')}")
    print("\n  Ctrl-C to stop\n")
    httpd.jobs.note(f"web UI started at {url} ({now_iso()})")

    if open_browser:
        import webbrowser

        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping…")
    finally:
        httpd.shutdown()
        httpd.server_close()
