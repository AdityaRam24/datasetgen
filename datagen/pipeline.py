"""The build pipeline.

    sources -> documents -> chunks -> dedupe -> generate -> quality -> persist

Incremental by default: a document whose content hash is unchanged since the
last run is skipped entirely, and chunks that already have records are not
regenerated. `--full` forces everything.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunking import chunk_document
from .config import Config
from .connectors import CONNECTORS, source_blocks
from .dedupe import Deduper, normalize_for_hash, simhash
from .exporters import export_all
from .generators import generate_all
from .llm import LocalLLM
from .models import Chunk, Document, Record, RunStats
from .quality import evaluate
from .state import StateStore
from .util import chunked, get_logger, now_iso, sha256, short_id

log = get_logger("pipeline")


def _chunk_row(chunk: Chunk) -> dict:
    row = asdict(chunk)
    if not chunk.embedding:
        row.pop("embedding", None)   # keep the file small when vectors are off
    return row


class Pipeline:
    def __init__(self, cfg: Config, state: StateStore, llm: LocalLLM) -> None:
        self.cfg = cfg
        self.state = state
        self.llm = llm

    # -- stages -------------------------------------------------------------

    def collect(self, only: list[str] | None = None) -> list[Document]:
        """Run every enabled connector."""
        docs: list[Document] = []
        blocks = source_blocks(self.cfg.sources)
        if only:
            blocks = [(t, b) for t, b in blocks if t in only or b.get("name") in only]

        if not blocks:
            log.warning("no enabled sources — check [[sources.*]] in config.toml")
            return []

        for type_, block in blocks:
            name = block.get("name", type_)
            log.info("── source: %s (%s) ──", name, type_)
            try:
                docs.extend(CONNECTORS[type_](self.cfg, block, self.state))
            except Exception as e:
                log.error("source %s failed: %s", name, e, exc_info=log.isEnabledFor(10))
        return docs

    def to_chunks(self, docs: list[Document], stats: RunStats, full: bool) -> list[Chunk]:
        """Persist document state and chunk the ones that changed."""
        chunks: list[Chunk] = []

        for doc in docs:
            stats.docs_fetched += 1
            changed = self.state.upsert_document(doc)
            processed = self.state.doc_has_chunks(doc.id)
            if not changed and processed and not full and self.cfg.generation.skip_if_unchanged:
                stats.docs_skipped_unchanged += 1
                log.debug("unchanged, skipping: %s", doc.title[:60])
                continue
            if changed:
                self.state.delete_chunks_for_doc(doc.id)  # old chunks are stale
            chunks.extend(chunk_document(doc, self.cfg.chunking))

        stats.chunks = len(chunks)
        log.info(
            "%d documents -> %d chunks (%d unchanged documents skipped)",
            len(docs), len(chunks), stats.docs_skipped_unchanged,
        )
        return chunks

    def deduplicate(self, chunks: list[Chunk], stats: RunStats, full: bool) -> list[Chunk]:
        d = self.cfg.dedupe
        deduper = Deduper(
            max_distance=d.simhash_bits,
            semantic_threshold=d.semantic_threshold,
            use_exact=d.exact,
            use_near=d.near,
            use_semantic=d.semantic and self.cfg.llm.embed_enabled,
        )
        if not full:
            # Dedupe against history, not just within this batch.
            deduper.seed(self.state.all_simhashes())

        embeddings = self.embed_chunks(chunks) if deduper.use_semantic else {}

        kept: list[Chunk] = []
        for chunk in chunks:
            dup = deduper.check(chunk.id, chunk.text, embeddings.get(chunk.id))
            if dup:
                continue
            chunk.embedding = embeddings.get(chunk.id)
            kept.append(chunk)
            # NB: chunks are registered in state only once their records are on
            # disk (see persist). Registering here would mark the work as done
            # before it was done: a crash or a Ctrl-C during generation would
            # leave the chunk recorded with no records, and every later
            # incremental run would skip its document as "already processed".

        stats.duplicates_exact = deduper.stats.exact
        stats.duplicates_near = deduper.stats.near
        stats.duplicates_semantic = deduper.stats.semantic
        stats.chunks_new = len(kept)
        log.info(
            "dedupe: kept %d/%d chunks (exact %d, near %d, semantic %d)",
            len(kept), len(chunks), deduper.stats.exact,
            deduper.stats.near, deduper.stats.semantic,
        )
        return kept

    def embed_chunks(self, chunks: list[Chunk]) -> dict[str, list[float]]:
        if not chunks or not self.llm.available():
            return {}
        log.info("embedding %d chunks with %s", len(chunks), self.cfg.llm.embed_model)
        out: dict[str, list[float]] = {}
        for batch in chunked(chunks, 32):
            vectors = self.llm.embed([c.text for c in batch])
            if vectors is None:
                log.info("embeddings unavailable — semantic dedupe disabled for this run")
                return out
            for chunk, vec in zip(batch, vectors):
                if vec:
                    out[chunk.id] = vec
        return out

    # -- persistence --------------------------------------------------------

    def _append_jsonl(self, path: Path, rows: list[dict], replace: bool) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w" if replace else "a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _load_existing_records(self) -> list[Record]:
        if not self.cfg.records_path.exists():
            return []
        out = []
        with open(self.cfg.records_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    out.append(
                        Record(**{k: v for k, v in data.items() if k in Record.__annotations__})
                    )
                except (json.JSONDecodeError, TypeError):
                    continue
        return out

    def persist(
        self,
        chunks: list[Chunk],
        accepted: list[Record],
        quarantined: list[Record],
        full: bool = False,
    ) -> list[Record]:
        """Write chunks and records to disk and register them in state.

        Every path that generates records goes through here — the pipeline, the
        CLI one-shot commands and the agent — so `datagen export` always sees
        the complete dataset regardless of how it was built.

        Returns every record now on disk (existing + newly added).
        """
        existing = [] if full else self._load_existing_records()
        known = {r.id for r in existing}
        fresh = [r for r in accepted if r.id not in known]

        self._append_jsonl(
            self.cfg.records_path, [r.to_dict() for r in existing + fresh], replace=True
        )
        self._append_jsonl(
            self.cfg.data_dir / "quarantine.jsonl",
            [r.to_dict() for r in quarantined],
            replace=full,
        )
        self._append_jsonl(
            self.cfg.chunks_path, [_chunk_row(c) for c in chunks], replace=full
        )
        # Only now is the work durable, so only now do the chunks count as
        # processed. Ordering matters: records first, then chunks — the reverse
        # would reintroduce the "processed but empty" state after a crash.
        self.state.add_records(fresh)
        for chunk in chunks:
            self.state.add_chunk(
                chunk.id, chunk.doc_id, sha256(normalize_for_hash(chunk.text)), simhash(chunk.text)
            )

        log.info(
            "persisted: +%d new records (%d total), %d quarantined, %d chunks",
            len(fresh), len(existing) + len(fresh), len(quarantined), len(chunks),
        )
        return existing + fresh

    # -- the run ------------------------------------------------------------

    def run(
        self,
        *,
        full: bool = False,
        only: list[str] | None = None,
        export: bool = True,
        limit: int = 0,
    ) -> RunStats:
        stats = RunStats(run_id=short_id(now_iso(), "run"))
        self.cfg.ensure_dirs()
        self.state.start_run(stats.run_id, "full" if full else "incremental")

        banner = (
            f"LOCAL model {self.cfg.llm.model} @ {self.cfg.llm.base_url}"
            if self.llm.available()
            else "NO local model reachable — extractive fallback"
        )
        log.info("=== build %s (%s) | %s ===", stats.run_id, "full" if full else "incremental", banner)

        try:
            docs = self.collect(only)
            chunks = self.to_chunks(docs, stats, full)

            if limit:
                chunks = chunks[:limit]
                log.info("limited to %d chunks", len(chunks))

            chunks = self.deduplicate(chunks, stats, full)

            if not chunks:
                log.info("nothing new to generate from — the dataset is already current")
                self.state.finish_run(stats.run_id, asdict(stats))
                if export:
                    export_all(self.cfg)
                return stats

            records = generate_all(chunks, self.llm, self.cfg, stats)

            deduper = Deduper()
            records = deduper.dedupe_records(records)

            accepted, quarantined = evaluate(records, self.llm, self.cfg.quality)
            stats.records = len(accepted)
            stats.records_quarantined = len(quarantined)

            all_records = self.persist(chunks, accepted, quarantined, full)

            if export:
                # chunks=None so the exporter reads the accumulated corpus from
                # disk — this run's new chunks are only part of it.
                export_all(self.cfg, all_records, None)

            stats.finished_at = now_iso()
            self.state.finish_run(stats.run_id, asdict(stats), ok=True)
            log.info("=== done: %s ===", stats.summary())
            return stats

        except KeyboardInterrupt:
            stats.errors.append("interrupted")
            self.state.finish_run(stats.run_id, asdict(stats), ok=False)
            raise
        except Exception as e:
            stats.errors.append(str(e))
            stats.finished_at = now_iso()
            self.state.finish_run(stats.run_id, asdict(stats), ok=False)
            log.error("build failed: %s", e, exc_info=True)
            raise


def build(cfg: Config, **kw: Any) -> RunStats:
    """Convenience wrapper used by the CLI and the scheduler."""
    llm = LocalLLM(cfg.llm)
    with StateStore(cfg.state_db) as state:
        return Pipeline(cfg, state, llm).run(**kw)
