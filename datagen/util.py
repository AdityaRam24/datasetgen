"""Shared helpers: logging, HTTP, hashing, text normalisation.

Deliberately stdlib-only — the whole generator must be runnable on a fresh
Python install with no `pip install` step.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Any, Iterable

USER_AGENT = "kalam-datagen/1.0 (+local dataset builder)"

# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

_LEVEL_COLORS = {
    "DEBUG": "\033[90m",
    "INFO": "\033[36m",
    "WARNING": "\033[33m",
    "ERROR": "\033[31m",
    "CRITICAL": "\033[41m",
}


class _Formatter(logging.Formatter):
    def __init__(self, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)-18s %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        out = super().format(record)
        if self.color:
            c = _LEVEL_COLORS.get(record.levelname, "")
            if c:
                out = out.replace(record.levelname, f"{c}{record.levelname}\033[0m", 1)
        return out


def force_utf8_output() -> None:
    """Windows consoles default to cp1252 and blow up on the box-drawing and
    check-mark characters used in the CLI output. Reconfigure rather than
    downgrade the output."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):
            pass


def setup_logging(verbose: bool = False, logfile: str | None = None) -> None:
    force_utf8_output()
    root = logging.getLogger("datagen")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()

    stream = logging.StreamHandler(sys.stderr)
    tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    color = tty and (os.name != "nt" or _win_ansi())
    stream.setFormatter(_Formatter(color))
    root.addHandler(stream)

    if logfile:
        os.makedirs(os.path.dirname(logfile) or ".", exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        root.addHandler(fh)


def _win_ansi() -> bool:
    """Windows 10+ terminals support ANSI once VT processing is enabled."""
    if os.name != "nt":
        return False
    try:
        import ctypes

        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"datagen.{name}")


log = get_logger("util")


# --------------------------------------------------------------------------
# Hashing / ids
# --------------------------------------------------------------------------


def sha256(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    return hashlib.sha256(data).hexdigest()


def short_id(*parts: Any, length: int = 16) -> str:
    return sha256("\x1f".join(str(p) for p in parts))[:length]


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

_WS = re.compile(r"[ \t\x0b\f\r]+")
_NL = re.compile(r"\n{3,}")
_CTRL = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")
_TOKEN = re.compile(r"[a-z0-9_.\-/]+")

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "be", "with", "as", "at", "by", "it", "this", "that", "from", "you", "your",
    "can", "will", "if", "not", "but", "has", "have", "was", "were", "we", "i",
}


def clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters, preserving paragraphs."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub(" ", text)
    text = _WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL.sub("\n\n", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercased content tokens. Mirrors the tokenizer in server/pcai/store.ts
    so lexical scores are comparable across the TS and Python sides."""
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1 and t not in _STOP]


def truncate(text: str, limit: int, suffix: str = " …") -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    sp = cut.rfind(" ")
    if sp > limit * 0.6:
        cut = cut[:sp]
    return cut + suffix


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 chars/token) — good enough for budgeting prompts."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# JSON coaxing — small local models fence their JSON or add prose around it
# --------------------------------------------------------------------------


def extract_json(raw: str) -> Any | None:
    """Pull the first valid JSON object/array out of a model response."""
    if not raw:
        return None
    text = raw.strip()

    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Scan for the first balanced {...} or [...] block.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
            start = text.find(opener, start + 1)
    return None


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


@dataclass
class Response:
    url: str
    status: int
    body: bytes
    headers: dict[str, str]

    @property
    def text(self) -> str:
        ctype = self.headers.get("content-type", "")
        m = re.search(r"charset=([\w\-]+)", ctype, re.I)
        for enc in ([m.group(1)] if m else []) + ["utf-8", "cp1252", "latin-1"]:
            try:
                return self.body.decode(enc)
            except (UnicodeDecodeError, LookupError):
                continue
        return self.body.decode("utf-8", "replace")

    def json(self) -> Any:
        return json.loads(self.text)

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()


class HttpError(Exception):
    def __init__(self, url: str, status: int, message: str) -> None:
        super().__init__(f"{status} {message} <{url}>")
        self.url, self.status = url, status


def http_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | str | None = None,
    timeout: float = 30.0,
    retries: int = 2,
    backoff: float = 1.5,
    max_bytes: int = 12 * 1024 * 1024,
) -> Response:
    """Minimal HTTP client with retries, gzip and a size ceiling.

    Raises HttpError on non-2xx after retries; transport errors surface as
    HttpError with status 0 so callers only handle one exception type.
    """
    hdrs = {
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip",
        "Accept": "*/*",
        **(headers or {}),
    }
    if isinstance(data, str):
        data = data.encode("utf-8")

    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(max_bytes + 1)
                if len(raw) > max_bytes:
                    log.warning("truncating oversized response from %s", url)
                    raw = raw[:max_bytes]
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    try:
                        raw = gzip.decompress(raw)
                    except (OSError, zlib.error):
                        pass  # some servers mislabel; keep the raw bytes
                return Response(
                    url=resp.geturl(),
                    status=resp.status,
                    body=raw,
                    headers={k.lower(): v for k, v in resp.headers.items()},
                )
        except urllib.error.HTTPError as e:
            body = b""
            try:
                body = e.read(4096)
            except Exception:
                pass
            # 4xx (except 429) will not improve on retry.
            if e.code < 500 and e.code != 429:
                raise HttpError(url, e.code, body.decode("utf-8", "replace")[:300] or e.reason)
            last = e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
        if attempt < retries:
            time.sleep(backoff ** (attempt + 1))

    raise HttpError(url, 0, str(last))


def http_json(url: str, **kw: Any) -> Any:
    kw.setdefault("headers", {})
    kw["headers"].setdefault("Accept", "application/json")
    return http_request(url, **kw).json()


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------


def chunked(items: Iterable[Any], size: int) -> Iterable[list[Any]]:
    buf: list[Any] = []
    for it in items:
        buf.append(it)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_dotenv(path: str) -> None:
    """Load KEY=VALUE lines into os.environ without overwriting real env vars."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n //= 1024
    return f"{n}TB"


def zip_member_text(data: bytes, pattern: str) -> list[str]:
    """Read text out of OOXML members matching `pattern` — the fallback path for
    docx/pptx/xlsx when the dedicated library is not installed."""
    import zipfile

    out: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if re.match(pattern, name):
                    out.append(zf.read(name).decode("utf-8", "replace"))
    except (zipfile.BadZipFile, KeyError, OSError) as e:
        log.debug("ooxml read failed: %s", e)
    return out
