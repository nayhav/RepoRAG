"""BM25 sparse index for keyword-based retrieval."""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from gitrag.core.types import CodeChunk

logger = logging.getLogger(__name__)

# Regex helpers for token splitting
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^a-zA-Z0-9_]")


class BM25Store:
    """In-memory BM25 index over code chunks."""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into search tokens.

        * Splits on whitespace / punctuation
        * Expands camelCase and snake_case
        * Lowercases everything
        * Drops tokens shorter than 2 chars
        """
        raw_tokens = _NON_ALNUM.split(text)
        tokens: list[str] = []
        for tok in raw_tokens:
            # Split snake_case
            parts = tok.split("_")
            for part in parts:
                # Split camelCase
                sub_parts = _CAMEL_BOUNDARY.split(part)
                for sp in sub_parts:
                    low = sp.lower()
                    if len(low) >= 2:
                        tokens.append(low)
        return tokens

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, chunks: list[CodeChunk]) -> None:
        """Build BM25 index from a list of code chunks."""
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._corpus = [self._tokenize(c.to_index_text()) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus)
        logger.info("BM25 index built with %d documents", len(self._chunk_ids))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_text: str, top_k: int) -> list[tuple[str, float]]:
        """Return ``(chunk_id, bm25_score)`` tuples sorted by score descending."""
        if self._bm25 is None or not self._chunk_ids:
            return []

        tokenized_query = self._tokenize(query_text)
        if not tokenized_query:
            return []

        scores: np.ndarray = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self._chunk_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Pickle the index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "chunk_ids": self._chunk_ids,
                    "corpus": self._corpus,
                },
                f,
            )
        logger.info("BM25 index saved to %s", path)

    def load(self, path: Path) -> None:
        """Load a previously saved index."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        self._chunk_ids = data["chunk_ids"]
        self._corpus = data["corpus"]
        self._bm25 = BM25Okapi(self._corpus)
        logger.info("BM25 index loaded from %s (%d documents)", path, len(self._chunk_ids))
