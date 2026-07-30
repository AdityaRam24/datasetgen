"""SQLite state store — this is what makes the generator incremental and
self-updating.

It remembers:
  * every document it has seen and that document's content hash
  * every chunk hash + simhash (so re-runs skip unchanged material)
  * every generated record id (so nothing is regenerated needlessly)
  * discovered/proposed sources the agent wants to visit next time
  * run history, for `datagen status`
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

from .util import get_logger, now_iso

log = get_logger("state")

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    title        TEXT,
    kind         TEXT,
    source       TEXT,
    content_hash TEXT,
    first_seen   TEXT,
    last_seen    TEXT,
    last_changed TEXT,
    revisions    INTEGER DEFAULT 1,
    meta         TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);

CREATE TABLE IF NOT EXISTS chunks (
    id       TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL,
    hash     TEXT NOT NULL,
    simhash  INTEGER,
    created  TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(hash);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS records (
    id       TEXT PRIMARY KEY,
    chunk_id TEXT,
    kind     TEXT,
    score    REAL,
    created  TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_chunk ON records(chunk_id);

-- Sources the agent discovered or proposed. `status`: proposed|accepted|rejected|done
CREATE TABLE IF NOT EXISTS discovered (
    key       TEXT PRIMARY KEY,   -- url or keyword
    type      TEXT NOT NULL,      -- url | keyword
    reason    TEXT,
    status    TEXT DEFAULT 'proposed',
    score     REAL DEFAULT 0,
    added_at  TEXT,
    used_at   TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id     TEXT PRIMARY KEY,
    started_at TEXT,
    finished_at TEXT,
    mode       TEXT,
    stats      TEXT,
    ok         INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# SQLite INTEGER is signed 64-bit; our simhashes are unsigned 64-bit. Round-trip
# through the signed range rather than storing them as strings, so the column
# stays indexable and comparable.
_SIGN_BIT = 1 << 63
_WRAP = 1 << 64


def _to_signed64(value: int) -> int:
    value &= _WRAP - 1
    return value - _WRAP if value >= _SIGN_BIT else value


def _to_unsigned64(value: int) -> int:
    return value + _WRAP if value < 0 else value


class StateStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self.db.executescript(SCHEMA)
            self.db.commit()

    def close(self) -> None:
        with self._lock:
            self.db.commit()
            self.db.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- documents ----------------------------------------------------------

    def document_changed(self, doc_id: str, content_hash: str) -> bool:
        """True when this document is new or its content differs from last run."""
        with self._lock:
            row = self.db.execute(
                "SELECT content_hash FROM documents WHERE id = ?", (doc_id,)
            ).fetchone()
        return row is None or row["content_hash"] != content_hash

    def upsert_document(self, doc: Any) -> bool:
        """Record a document; returns True if its content changed since last time."""
        changed = self.document_changed(doc.id, doc.content_hash)
        ts = now_iso()
        with self._lock:
            if changed:
                self.db.execute(
                    """INSERT INTO documents
                         (id, url, title, kind, source, content_hash,
                          first_seen, last_seen, last_changed, revisions, meta)
                       VALUES (?,?,?,?,?,?,?,?,?,1,?)
                       ON CONFLICT(id) DO UPDATE SET
                         content_hash = excluded.content_hash,
                         title        = excluded.title,
                         last_seen    = excluded.last_seen,
                         last_changed = excluded.last_changed,
                         revisions    = documents.revisions + 1,
                         meta         = excluded.meta""",
                    (
                        doc.id, doc.url, doc.title, doc.kind, doc.source, doc.content_hash,
                        ts, ts, ts, json.dumps(doc.meta or {}),
                    ),
                )
            else:
                self.db.execute(
                    "UPDATE documents SET last_seen = ? WHERE id = ?", (ts, doc.id)
                )
            self.db.commit()
        return changed

    def known_urls(self, source: str | None = None) -> set[str]:
        q = "SELECT url FROM documents" + (" WHERE source = ?" if source else "")
        with self._lock:
            rows = self.db.execute(q, (source,) if source else ()).fetchall()
        return {r["url"] for r in rows}

    # -- chunks -------------------------------------------------------------

    def chunk_seen(self, chunk_hash: str) -> bool:
        with self._lock:
            return (
                self.db.execute(
                    "SELECT 1 FROM chunks WHERE hash = ? LIMIT 1", (chunk_hash,)
                ).fetchone()
                is not None
            )

    def add_chunk(self, chunk_id: str, doc_id: str, chunk_hash: str, simhash: int) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO chunks (id, doc_id, hash, simhash, created) "
                "VALUES (?,?,?,?,?)",
                (chunk_id, doc_id, chunk_hash, _to_signed64(simhash), now_iso()),
            )
            self.db.commit()

    def doc_has_chunks(self, doc_id: str) -> bool:
        """A document is only really "processed" once its chunks exist. Without
        this check, a run that dies between upsert_document and chunking would
        make the document look unchanged forever and it would never be
        reprocessed."""
        with self._lock:
            return (
                self.db.execute(
                    "SELECT 1 FROM chunks WHERE doc_id = ? LIMIT 1", (doc_id,)
                ).fetchone()
                is not None
            )

    def all_simhashes(self) -> list[tuple[str, int]]:
        with self._lock:
            rows = self.db.execute(
                "SELECT id, simhash FROM chunks WHERE simhash IS NOT NULL"
            ).fetchall()
        return [(r["id"], _to_unsigned64(r["simhash"])) for r in rows]

    def delete_chunks_for_doc(self, doc_id: str) -> None:
        """Called when a document changed — its old chunks are stale."""
        with self._lock:
            self.db.execute(
                "DELETE FROM records WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self.db.commit()

    # -- records ------------------------------------------------------------

    def record_exists(self, record_id: str) -> bool:
        with self._lock:
            return (
                self.db.execute("SELECT 1 FROM records WHERE id = ?", (record_id,)).fetchone()
                is not None
            )

    def add_records(self, records: Iterable[Any]) -> None:
        rows = [(r.id, r.chunk_id, r.kind, r.score, now_iso()) for r in records]
        if not rows:
            return
        with self._lock:
            self.db.executemany(
                "INSERT OR REPLACE INTO records (id, chunk_id, kind, score, created) "
                "VALUES (?,?,?,?,?)",
                rows,
            )
            self.db.commit()

    def chunks_without_records(self) -> set[str]:
        with self._lock:
            rows = self.db.execute(
                "SELECT c.id FROM chunks c LEFT JOIN records r ON r.chunk_id = c.id "
                "WHERE r.id IS NULL"
            ).fetchall()
        return {r["id"] for r in rows}

    # -- discovered sources (self-updating behaviour) -----------------------

    def propose(self, key: str, type_: str, reason: str = "", score: float = 0.0) -> bool:
        """Register a URL/keyword the agent wants to explore. Returns False if
        it was already known (proposed, used or rejected)."""
        with self._lock:
            existing = self.db.execute("SELECT 1 FROM discovered WHERE key = ?", (key,)).fetchone()
            if existing:
                return False
            self.db.execute(
                "INSERT INTO discovered (key, type, reason, status, score, added_at) "
                "VALUES (?,?,?,'proposed',?,?)",
                (key, type_, reason, score, now_iso()),
            )
            self.db.commit()
        return True

    def pending(self, type_: str | None = None, limit: int = 50) -> list[dict]:
        q = "SELECT * FROM discovered WHERE status = 'proposed'"
        args: list[Any] = []
        if type_:
            q += " AND type = ?"
            args.append(type_)
        q += " ORDER BY score DESC, added_at ASC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [dict(r) for r in self.db.execute(q, args).fetchall()]

    def mark(self, key: str, status: str) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE discovered SET status = ?, used_at = ? WHERE key = ?",
                (status, now_iso(), key),
            )
            self.db.commit()

    # -- runs & kv ----------------------------------------------------------

    def start_run(self, run_id: str, mode: str) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO runs (run_id, started_at, mode, ok) VALUES (?,?,?,0)",
                (run_id, now_iso(), mode),
            )
            self.db.commit()

    def finish_run(self, run_id: str, stats: dict, ok: bool = True) -> None:
        with self._lock:
            self.db.execute(
                "UPDATE runs SET finished_at = ?, stats = ?, ok = ? WHERE run_id = ?",
                (now_iso(), json.dumps(stats, default=str), 1 if ok else 0, run_id),
            )
            self.db.commit()

    def recent_runs(self, limit: int = 10) -> list[dict]:
        with self._lock:
            rows = self.db.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self.db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.db.execute(
                "INSERT OR REPLACE INTO kv (key, value) VALUES (?,?)",
                (key, json.dumps(value, default=str)),
            )
            self.db.commit()

    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {}
            for table in ("documents", "chunks", "records", "discovered", "runs"):
                out[table] = self.db.execute(f"SELECT COUNT(*) c FROM {table}").fetchone()["c"]
        return out
