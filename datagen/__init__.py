"""datagen — an agentic dataset generator driven by a LOCAL LLM.

Reads PDFs/Office docs, scrapes the web by keyword, pulls runbooks and
Confluence pages, and turns all of it into a deduplicated, quality-gated,
citation-carrying training/RAG dataset. It runs itself on a schedule and
proposes its own next sources.

Nothing is sent to a hosted model: generation, judging and embeddings all go
to a local Ollama / OpenAI-compatible server on your machine.

    from datagen import load_config, build
    cfg = load_config()
    build(cfg)
"""

from .config import Config, load_config
from .exporters import export_all
from .llm import LocalLLM
from .models import Chunk, Document, Record, RunStats
from .pipeline import Pipeline, build
from .state import StateStore

__version__ = "1.0.0"

__all__ = [
    "Config",
    "load_config",
    "build",
    "Pipeline",
    "StateStore",
    "LocalLLM",
    "Document",
    "Chunk",
    "Record",
    "RunStats",
    "export_all",
    "__version__",
]
