"""Three-layer deduplication.

1. exact     — sha256 of normalised text
2. near      — 64-bit SimHash over token shingles, Hamming distance threshold.
               Catches boilerplate headers/footers and pages that differ only by
               a version number, which web crawls produce in bulk.
3. semantic  — cosine similarity over local embeddings, for paraphrases.

Dropping duplicates is the single highest-leverage quality step: duplicated
chunks turn into duplicated training rows, which is how a fine-tune memorises
noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .util import get_logger, sha256

log = get_logger("dedupe")

_WORD = re.compile(r"[a-z0-9]+")
_MASK64 = (1 << 64) - 1


def normalize_for_hash(text: str) -> str:
    """Aggressive normalisation used only for equality testing."""
    return " ".join(_WORD.findall(text.lower()))


def simhash(text: str, shingle: int = 3) -> int:
    """64-bit SimHash over word shingles."""
    words = _WORD.findall(text.lower())
    if not words:
        return 0
    grams = (
        [" ".join(words[i : i + shingle]) for i in range(len(words) - shingle + 1)]
        if len(words) >= shingle
        else [" ".join(words)]
    )
    vector = [0] * 64
    for gram in grams:
        h = int(sha256(gram)[:16], 16)
        for bit in range(64):
            vector[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if vector[bit] > 0:
            out |= 1 << bit
    return out & _MASK64


def hamming(a: int, b: int) -> int:
    return bin((a ^ b) & _MASK64).count("1")


class _SimhashIndex:
    """Banded index: split the 64-bit hash into 8 bands of 8 bits. Two hashes
    within distance <= 7 must share at least one band, so we only compare
    candidates instead of doing an O(n²) sweep."""

    BANDS = 8
    WIDTH = 8

    def __init__(self) -> None:
        self.buckets: dict[tuple[int, int], list[tuple[str, int]]] = {}

    def _keys(self, h: int):
        for band in range(self.BANDS):
            yield (band, (h >> (band * self.WIDTH)) & ((1 << self.WIDTH) - 1))

    def add(self, key: str, h: int) -> None:
        for k in self._keys(h):
            self.buckets.setdefault(k, []).append((key, h))

    def query(self, h: int, max_distance: int) -> str | None:
        seen: set[str] = set()
        for k in self._keys(h):
            for key, other in self.buckets.get(k, ()):
                if key in seen:
                    continue
                seen.add(key)
                if hamming(h, other) <= max_distance:
                    return key
        return None


@dataclass
class DedupeStats:
    exact: int = 0
    near: int = 0
    semantic: int = 0

    @property
    def total(self) -> int:
        return self.exact + self.near + self.semantic


@dataclass
class Deduper:
    """Stateful across a run; seed it with hashes from previous runs to make
    incremental runs deduplicate against history too."""

    max_distance: int = 6
    semantic_threshold: float = 0.94
    use_exact: bool = True
    use_near: bool = True
    use_semantic: bool = True

    stats: DedupeStats = field(default_factory=DedupeStats)
    _hashes: set[str] = field(default_factory=set)
    _index: _SimhashIndex = field(default_factory=_SimhashIndex)
    _vectors: list[tuple[str, list[float]]] = field(default_factory=list)

    def seed(self, simhashes: list[tuple[str, int]]) -> None:
        for key, h in simhashes:
            if h:
                self._index.add(key, h)

    def check(self, key: str, text: str, embedding: list[float] | None = None) -> str | None:
        """Return the id of the chunk this duplicates, or None if it is novel.
        Novel items are registered so later items dedupe against them."""
        h = sha256(normalize_for_hash(text))
        if self.use_exact:
            if h in self._hashes:
                self.stats.exact += 1
                return "exact"
            self._hashes.add(h)

        if self.use_near:
            sh = simhash(text)
            if sh:
                hit = self._index.query(sh, self.max_distance)
                if hit and hit != key:
                    self.stats.near += 1
                    return hit
                self._index.add(key, sh)

        if self.use_semantic and embedding:
            from .llm import cosine

            for other_key, vec in self._vectors:
                if cosine(embedding, vec) >= self.semantic_threshold:
                    self.stats.semantic += 1
                    return other_key
            self._vectors.append((key, embedding))

        return None

    def dedupe_records(self, records: list) -> list:
        """Drop records whose *question* is a near-duplicate of another's.
        Two chunks describing the same thing often yield the same question."""
        seen: dict[str, str] = {}
        index = _SimhashIndex()
        out = []
        for rec in records:
            norm = normalize_for_hash(rec.instruction)
            if not norm:
                continue
            if norm in seen:
                continue
            sh = simhash(rec.instruction)
            if sh and index.query(sh, 3):
                continue
            seen[norm] = rec.id
            if sh:
                index.add(rec.id, sh)
            out.append(rec)
        dropped = len(records) - len(out)
        if dropped:
            log.info("dropped %d duplicate questions", dropped)
        return out
