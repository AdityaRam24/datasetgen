"""Local LLM client.

THIS PROJECT USES A LOCAL MODEL. There is no cloud provider here and adding one
is not a supported configuration — every prompt, every chunk of your runbooks
and Confluence pages stays on your machine.

Supported backends:
  * ollama         — native /api/chat + /api/embeddings  (default)
  * openai_compat  — /v1/chat/completions + /v1/embeddings
                     (LM Studio, llama.cpp server, vLLM, Ollama's own /v1)
  * none           — disables the LLM; the pipeline runs extractive-only

Everything degrades instead of exploding: if the daemon is down, `complete()`
returns None and callers fall back to deterministic extraction.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .config import LLMConfig
from .util import HttpError, extract_json, get_logger, http_request

log = get_logger("llm")


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    prompt_chars: int = 0
    completion_chars: int = 0
    seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, ok: bool, pchars: int, cchars: int, secs: float) -> None:
        with self._lock:
            self.calls += 1
            if not ok:
                self.failures += 1
            self.prompt_chars += pchars
            self.completion_chars += cchars
            self.seconds += secs


class LocalLLM:
    """Thin, thread-safe wrapper around a locally hosted model."""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.stats = LLMStats()
        self._available: bool | None = None
        self._embed_checked = False
        self._lock = threading.Lock()

    # -- health -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.cfg.provider != "none"

    def available(self, recheck: bool = False) -> bool:
        """Ping the daemon once and cache the answer."""
        if not self.enabled:
            return False
        with self._lock:
            if self._available is not None and not recheck:
                return self._available
            try:
                url = (
                    f"{self.cfg.base_url}/api/tags"
                    if self.cfg.provider == "ollama"
                    else f"{self.cfg.base_url}/v1/models"
                )
                http_request(url, timeout=8, retries=0)
                self._available = self._resolve_model()
                if self._available:
                    log.info(
                        "local LLM reachable at %s (model=%s)", self.cfg.base_url, self.cfg.model
                    )
            except HttpError as e:
                self._available = False
                log.warning(
                    "local LLM NOT reachable at %s (%s). "
                    "Start it with `ollama serve` — falling back to extractive generation.",
                    self.cfg.base_url,
                    e,
                )
            return self._available

    def _resolve_model(self) -> bool:
        """Verify the configured model is actually installed.

        A reachable daemon says nothing about the model: asking for one that was
        never pulled 404s on *every* call, and the run then spends half an hour
        producing extractive junk. Check once, substitute an installed chat model
        if we can, and give up loudly if we cannot.
        """
        installed = self.installed_models()
        if not installed:
            return True  # server reports nothing useful; let the call decide

        self._resolve_embed_model(installed)

        want = self.cfg.model
        if want in installed:
            return True

        # Tolerate tag drift: `qwen2.5:7b-instruct` vs `qwen2.5:7b-instruct-q4_K_M`,
        # and a bare name against the `:latest` tag.
        for name in installed:
            if name == f"{want}:latest" or name.startswith(want) or want.startswith(name):
                log.warning("model %r is not installed — using %r instead", want, name)
                self.cfg.model = name
                return True

        chat = [m for m in installed if not _looks_like_embedder(m, {})]
        if chat:
            pick = min(chat, key=lambda m: (_family_rank(m, want), m))
            log.warning(
                "model %r is not installed. Falling back to %r "
                "(installed: %s). Fix this in config.toml or run: ollama pull %s",
                want, pick, ", ".join(sorted(installed)), want,
            )
            self.cfg.model = pick
            return True

        log.error(
            "model %r is not installed and no chat model is available (installed: %s). "
            "Run: ollama pull %s — generation will be extractive-only until you do.",
            want, ", ".join(sorted(installed)) or "none", want,
        )
        return False

    def model_catalog(self) -> list[dict[str, Any]]:
        """Installed models with size and parameter count, for a picker UI.

        Ollama reports both; OpenAI-compatible servers usually report neither,
        so those fields come back empty rather than guessed.
        """
        try:
            if self.cfg.provider == "ollama":
                data = http_request(f"{self.cfg.base_url}/api/tags", timeout=8, retries=0).json()
                out = []
                for m in data.get("models", []):
                    details = m.get("details") or {}
                    size = int(m.get("size") or 0)
                    out.append({
                        "name": m.get("name", ""),
                        "bytes": size,
                        "size": _human_gb(size),
                        "params": details.get("parameter_size", ""),
                        "family": details.get("family", ""),
                        "quantization": details.get("quantization_level", ""),
                        # Embedding models cannot chat, and chat models embed
                        # badly. Flagging it here stops the obvious mis-pick.
                        "embedding": _looks_like_embedder(m.get("name", ""), details),
                    })
                return sorted(out, key=lambda m: m["name"])

            data = http_request(f"{self.cfg.base_url}/v1/models", timeout=8, retries=0).json()
            return [
                {"name": m.get("id", ""), "bytes": 0, "size": "", "params": "",
                 "family": "", "quantization": "",
                 "embedding": _looks_like_embedder(m.get("id", ""), {})}
                for m in data.get("data", [])
            ]
        except (HttpError, ValueError, KeyError, TypeError):
            return []

    def installed_models(self) -> list[str]:
        try:
            if self.cfg.provider == "ollama":
                data = http_request(f"{self.cfg.base_url}/api/tags", timeout=8, retries=0).json()
                return [m.get("name", "") for m in data.get("models", [])]
            data = http_request(f"{self.cfg.base_url}/v1/models", timeout=8, retries=0).json()
            return [m.get("id", "") for m in data.get("data", [])]
        except (HttpError, ValueError, KeyError):
            return []

    def _resolve_embed_model(self, installed: list[str]) -> None:
        """Same check for the embedding model, which fails separately.

        A missing embedder does not stop the run — it just silently disables
        semantic dedupe after failing once per batch. Substitute or switch it
        off deliberately, with one line saying so.
        """
        if not self.cfg.embed_enabled or self._embed_checked:
            return
        self._embed_checked = True

        want = self.cfg.embed_model
        if want in installed or any(m.startswith(want) for m in installed):
            return

        embedders = [m for m in installed if _looks_like_embedder(m, {})]
        if embedders:
            pick = sorted(embedders)[0]
            log.warning("embedding model %r is not installed — using %r", want, pick)
            self.cfg.embed_model = pick
            return

        log.warning(
            "embedding model %r is not installed and no embedder is available. "
            "Semantic near-duplicate detection is off for this run; exact and "
            "fuzzy dedupe still apply. Run: ollama pull %s",
            want, want,
        )
        self.cfg.embed_enabled = False

    # -- generation ---------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str | None:
        """Return the model's text, or None if the call failed.

        Never raises — callers treat None as "use the fallback path".
        """
        if not self.available():
            return None

        temp = self.cfg.temperature if temperature is None else temperature
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        started = time.time()

        for attempt in range(self.cfg.max_retries + 1):
            try:
                text = (
                    self._ollama_chat(messages, temp, max_tokens, json_mode)
                    if self.cfg.provider == "ollama"
                    else self._openai_chat(messages, temp, max_tokens, json_mode)
                )
                self.stats.record(True, len(prompt), len(text or ""), time.time() - started)
                return text
            except (HttpError, ValueError, KeyError) as e:
                # A missing model, a bad payload — retrying cannot fix these, and
                # retrying them once per chunk is how a run burns 20 minutes
                # logging the same 404. Disable the LLM for the rest of the run.
                if _is_fatal(e):
                    self.stats.record(False, len(prompt), 0, time.time() - started)
                    with self._lock:
                        if self._available is not False:
                            log.error(
                                "LLM disabled for this run: %s. "
                                "Falling back to extractive generation — fix the model "
                                "in config.toml (`python -m datagen doctor`) and re-run.",
                                e,
                            )
                        self._available = False
                    return None
                if attempt >= self.cfg.max_retries:
                    self.stats.record(False, len(prompt), 0, time.time() - started)
                    log.warning("LLM call failed after %d attempts: %s", attempt + 1, e)
                    with self._lock:
                        self._available = None  # re-probe next time; daemon may have died
                    return None
                time.sleep(1.5 * (attempt + 1))
        return None

    def complete_json(self, prompt: str, *, system: str | None = None, **kw: Any) -> Any | None:
        """Ask for JSON and parse it, tolerating fences and stray prose."""
        raw = self.complete(prompt, system=system, json_mode=True, **kw)
        if raw is None:
            return None
        parsed = extract_json(raw)
        if parsed is None:
            log.debug("model returned unparseable JSON: %s", raw[:300])
        return parsed

    def _ollama_chat(
        self, messages: list[dict], temp: float, max_tokens: int | None, json_mode: bool
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temp, "num_ctx": self.cfg.num_ctx},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        if json_mode:
            payload["format"] = "json"

        resp = http_request(
            f"{self.cfg.base_url}/api/chat",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=self.cfg.timeout,
            retries=0,
        )
        return (resp.json().get("message") or {}).get("content", "")

    def _openai_chat(
        self, messages: list[dict], temp: float, max_tokens: int | None, json_mode: bool
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temp,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = http_request(
            f"{self.cfg.base_url}/v1/chat/completions",
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
            data=json.dumps(payload),
            timeout=self.cfg.timeout,
            retries=0,
        )
        choices = resp.json().get("choices") or []
        return (choices[0].get("message") or {}).get("content", "") if choices else ""

    # -- embeddings ---------------------------------------------------------

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Embed a batch. Returns None if embeddings are unavailable — the
        caller then falls back to lexical/simhash similarity only."""
        if not texts or not self.cfg.embed_enabled or not self.available():
            return None
        try:
            if self.cfg.provider == "ollama":
                # /api/embed takes a batch; older builds only have /api/embeddings.
                try:
                    data = http_request(
                        f"{self.cfg.base_url}/api/embed",
                        method="POST",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({"model": self.cfg.embed_model, "input": texts}),
                        timeout=self.cfg.timeout,
                        retries=0,
                    ).json()
                    if data.get("embeddings"):
                        return data["embeddings"]
                except HttpError:
                    pass
                out = []
                for t in texts:
                    d = http_request(
                        f"{self.cfg.base_url}/api/embeddings",
                        method="POST",
                        headers={"Content-Type": "application/json"},
                        data=json.dumps({"model": self.cfg.embed_model, "prompt": t}),
                        timeout=self.cfg.timeout,
                        retries=0,
                    ).json()
                    out.append(d.get("embedding") or [])
                return out

            data = http_request(
                f"{self.cfg.base_url}/v1/embeddings",
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
                data=json.dumps({"model": self.cfg.embed_model, "input": texts}),
                timeout=self.cfg.timeout,
                retries=0,
            ).json()
            return [item["embedding"] for item in data.get("data", [])]
        except (HttpError, ValueError, KeyError, IndexError) as e:
            log.warning("embedding failed (%s) — continuing without vectors", e)
            return None


_FATAL_HINTS = ("not found", "no such model", "unknown model", "does not exist")


def _is_fatal(err: Exception) -> bool:
    """True for errors that will recur identically on every subsequent call."""
    if isinstance(err, HttpError) and err.status in (400, 401, 403, 404):
        return True
    return any(h in str(err).lower() for h in _FATAL_HINTS)


def _human_gb(n: int) -> str:
    if not n:
        return ""
    gb = n / (1024**3)
    return f"{gb:.1f} GB" if gb >= 1 else f"{n / (1024**2):.0f} MB"


_EMBED_HINTS = ("embed", "bge", "gte", "e5-", "minilm", "nomic-embed", "mxbai")

# Substitution order when the configured model is missing. These prompts ask for
# strict JSON, so a same-family instruct model is the closest stand-in; tiny and
# vision models are last because they mangle the schema.
_FAMILY_ORDER = ("qwen", "llama", "mistral", "mixtral", "gemma", "phi", "deepseek")
_WEAK_HINTS = ("moondream", "llava", "vision", ":1b", ":2b", "tinyllama")


def _family_rank(name: str, want: str) -> int:
    low = name.lower()
    if any(h in low for h in _WEAK_HINTS):
        return 90
    family = re.split(r"[:.\-]", want.lower())[0]
    if family and family in low:
        return 0
    for i, fam in enumerate(_FAMILY_ORDER):
        if fam in low:
            return 1 + i
    return 50


def _looks_like_embedder(name: str, details: dict) -> bool:
    low = f"{name} {details.get('family', '')}".lower()
    return any(h in low for h in _EMBED_HINTS)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na**0.5) * (nb**0.5))
