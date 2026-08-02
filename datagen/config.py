"""Configuration loading: config.toml + .env + CLI overrides.

TOML is read with the stdlib `tomllib` (3.11+) so there is no YAML dependency.
Every field has a default, so a missing/partial config.toml still runs.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import get_logger, load_dotenv

log = get_logger("config")


def _get(d: dict, path: str, default: Any = None) -> Any:
    """Dotted lookup: _get(cfg, 'llm.embeddings.model')."""
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


@dataclass
class LLMConfig:
    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b-instruct"
    temperature: float = 0.2
    num_ctx: int = 8192
    timeout: float = 180.0
    max_retries: int = 2
    allow_extractive_fallback: bool = True
    embed_enabled: bool = True
    embed_model: str = "nomic-embed-text"
    # Vision: screenshots and diagrams are described by a local vision model and
    # the description is treated as the document's text from then on.
    vision_enabled: bool = True
    vision_model: str = "gemma4"
    vision_max_bytes: int = 12 * 1024 * 1024

    @property
    def is_local(self) -> bool:
        """Guard rail: refuse to talk to anything that is not on this machine
        or a private network. This project is local-LLM only, by design."""
        host = self.base_url.split("://")[-1].split("/")[0].split(":")[0].lower()
        if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"):
            return True
        # RFC1918 / link-local — a local LLM box on the LAN is still local.
        return (
            host.startswith("10.")
            or host.startswith("192.168.")
            or host.startswith("169.254.")
            or any(host.startswith(f"172.{i}.") for i in range(16, 32))
            or host.endswith(".local")
            or host.endswith(".lan")
        )


@dataclass
class ChunkingConfig:
    max_chars: int = 1600
    overlap: int = 200
    min_chars: int = 220
    respect_headings: bool = True


@dataclass
class DedupeConfig:
    exact: bool = True
    near: bool = True
    simhash_bits: int = 6
    semantic: bool = True
    semantic_threshold: float = 0.94


@dataclass
class GenerationConfig:
    kinds: list[str] = field(default_factory=lambda: ["qa", "instruction", "troubleshooting"])
    pairs_per_chunk: int = 3
    max_workers: int = 4
    include_citations: bool = True
    skip_if_unchanged: bool = True
    # Write results every N chunks. An interrupted run then loses at most this
    # much work instead of everything since it started.
    checkpoint_every: int = 25


@dataclass
class QualityConfig:
    enabled: bool = True
    min_score: float = 0.6
    require_grounded: bool = True
    max_answer_chars: int = 2000
    min_question_chars: int = 12
    # "Which auth broker does PCAI use?" -> "Keycloak" is a perfect record. A
    # high floor here silently deletes exactly the crisp factual lookups an ops
    # assistant is asked for; grounding and the judge still police short answers.
    min_answer_chars: int = 12


@dataclass
class AgentConfig:
    enabled: bool = True
    max_steps: int = 24
    max_tool_errors: int = 5
    objective: str = "Build a high-quality, well-cited dataset from the configured sources."
    propose_new_sources: bool = True
    max_proposed_per_run: int = 10


@dataclass
class ScheduleConfig:
    enabled: bool = False
    interval_min: int = 360
    full_rebuild_every: int = 14
    jitter_pct: int = 10


@dataclass
class ExportConfig:
    formats: list[str] = field(default_factory=lambda: ["jsonl", "rag_chunks"])
    out_dir: str = "exports"
    train_split: float = 0.9
    shuffle_seed: int = 42
    kalam_kb_path: str = ""


@dataclass
class Config:
    root: Path
    name: str = "dataset"
    description: str = ""
    data_dir: Path = Path("data")
    language: str = "en"

    llm: LLMConfig = field(default_factory=LLMConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    dedupe: DedupeConfig = field(default_factory=DedupeConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Source blocks are kept as plain dicts — connectors know their own shape.
    sources: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    # -- derived paths ------------------------------------------------------
    @property
    def state_db(self) -> Path:
        return self.data_dir / "state.db"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def records_path(self) -> Path:
        return self.data_dir / "records.jsonl"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def export_dir(self) -> Path:
        p = Path(self.export.out_dir)
        return p if p.is_absolute() else self.root / p

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.cache_dir, self.export_dir):
            p.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(path).expanduser()
        return p if p.is_absolute() else (self.root / p)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg_path = Path(path) if path else Path(__file__).resolve().parent.parent / "config.toml"
    root = cfg_path.parent

    load_dotenv(str(root / ".env"))

    raw: dict[str, Any] = {}
    if cfg_path.is_file():
        with open(cfg_path, "rb") as fh:
            raw = tomllib.load(fh)
    else:
        log.warning("no config at %s — using defaults", cfg_path)

    llm = LLMConfig(
        provider=_get(raw, "llm.provider", "ollama"),
        base_url=os.getenv("DATAGEN_LLM_BASE_URL")
        or _get(raw, "llm.base_url", "http://localhost:11434"),
        model=os.getenv("DATAGEN_LLM_MODEL") or _get(raw, "llm.model", "qwen2.5:7b-instruct"),
        temperature=float(_get(raw, "llm.temperature", 0.2)),
        num_ctx=int(_get(raw, "llm.num_ctx", 8192)),
        timeout=float(_get(raw, "llm.timeout", 180)),
        max_retries=int(_get(raw, "llm.max_retries", 2)),
        allow_extractive_fallback=bool(_get(raw, "llm.allow_extractive_fallback", True)),
        embed_enabled=bool(_get(raw, "llm.embeddings.enabled", True)),
        embed_model=os.getenv("DATAGEN_EMBED_MODEL")
        or _get(raw, "llm.embeddings.model", "nomic-embed-text"),
        vision_enabled=bool(_get(raw, "llm.vision.enabled", True)),
        vision_model=os.getenv("DATAGEN_VISION_MODEL")
        or _get(raw, "llm.vision.model", "gemma4"),
        vision_max_bytes=int(_get(raw, "llm.vision.max_bytes", 12 * 1024 * 1024)),
    )

    split = _get(raw, "export.split", {}) or {}
    export = ExportConfig(
        formats=list(_get(raw, "export.formats", ["jsonl", "rag_chunks"])),
        out_dir=_get(raw, "export.out_dir", "exports"),
        train_split=float(split.get("train", 0.9)),
        shuffle_seed=int(_get(raw, "export.shuffle_seed", 42)),
        kalam_kb_path=_get(raw, "export.kalam_kb_path", ""),
    )

    data_dir = Path(_get(raw, "project.data_dir", "data"))
    cfg = Config(
        root=root,
        name=_get(raw, "project.name", "dataset"),
        description=_get(raw, "project.description", ""),
        data_dir=data_dir if data_dir.is_absolute() else root / data_dir,
        language=_get(raw, "project.language", "en"),
        llm=llm,
        chunking=ChunkingConfig(
            max_chars=int(_get(raw, "chunking.max_chars", 1600)),
            overlap=int(_get(raw, "chunking.overlap", 200)),
            min_chars=int(_get(raw, "chunking.min_chars", 220)),
            respect_headings=bool(_get(raw, "chunking.respect_headings", True)),
        ),
        dedupe=DedupeConfig(
            exact=bool(_get(raw, "dedupe.exact", True)),
            near=bool(_get(raw, "dedupe.near", True)),
            simhash_bits=int(_get(raw, "dedupe.simhash_bits", 6)),
            semantic=bool(_get(raw, "dedupe.semantic", True)),
            semantic_threshold=float(_get(raw, "dedupe.semantic_threshold", 0.94)),
        ),
        generation=GenerationConfig(
            kinds=list(_get(raw, "generation.kinds", ["qa"])),
            pairs_per_chunk=int(_get(raw, "generation.pairs_per_chunk", 3)),
            max_workers=int(_get(raw, "generation.max_workers", 4)),
            include_citations=bool(_get(raw, "generation.include_citations", True)),
            skip_if_unchanged=bool(_get(raw, "generation.skip_if_unchanged", True)),
            checkpoint_every=int(_get(raw, "generation.checkpoint_every", 25)),
        ),
        quality=QualityConfig(
            enabled=bool(_get(raw, "quality.enabled", True)),
            min_score=float(_get(raw, "quality.min_score", 0.6)),
            require_grounded=bool(_get(raw, "quality.require_grounded", True)),
            max_answer_chars=int(_get(raw, "quality.max_answer_chars", 2000)),
            min_question_chars=int(_get(raw, "quality.min_question_chars", 12)),
            min_answer_chars=int(_get(raw, "quality.min_answer_chars", 12)),
        ),
        agent=AgentConfig(
            enabled=bool(_get(raw, "agent.enabled", True)),
            max_steps=int(_get(raw, "agent.max_steps", 24)),
            max_tool_errors=int(_get(raw, "agent.max_tool_errors", 5)),
            objective=str(_get(raw, "agent.objective", "")).strip() or AgentConfig.objective,
            propose_new_sources=bool(_get(raw, "agent.propose_new_sources", True)),
            max_proposed_per_run=int(_get(raw, "agent.max_proposed_per_run", 10)),
        ),
        schedule=ScheduleConfig(
            enabled=bool(_get(raw, "schedule.enabled", False)),
            interval_min=int(_get(raw, "schedule.interval_min", 360)),
            full_rebuild_every=int(_get(raw, "schedule.full_rebuild_every", 14)),
            jitter_pct=int(_get(raw, "schedule.jitter_pct", 10)),
        ),
        export=export,
        sources=_get(raw, "sources", {}) or {},
        raw=raw,
    )

    if cfg.llm.provider != "none" and not cfg.llm.is_local:
        log.warning(
            "llm.base_url %s is not a local/private address. This project is "
            "designed for a LOCAL model; remote endpoints are not supported.",
            cfg.llm.base_url,
        )

    return cfg
