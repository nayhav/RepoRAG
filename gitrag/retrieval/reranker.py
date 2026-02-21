"""Cross-encoder reranking for retrieved code chunks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

    from gitrag.core.types import CodeChunk

logger = logging.getLogger(__name__)


class Reranker:
    """Lazy-loading cross-encoder reranker.

    The underlying ``sentence_transformers.CrossEncoder`` model is loaded on
    the first call to :meth:`rerank` so that import time stays fast and GPU
    memory is only consumed when reranking is actually needed.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        chunks: list[CodeChunk],
        top_k: int,
    ) -> list[tuple[CodeChunk, float]]:
        """Score ``(query, chunk)`` pairs and return the top-k by score.

        Parameters
        ----------
        query:
            The user query.
        chunks:
            Candidate chunks to rerank.
        top_k:
            Number of results to return.

        Returns
        -------
        list[tuple[CodeChunk, float]]
            Chunks paired with their reranker scores, sorted descending.
        """
        if not chunks:
            return []

        model = self._load_model()

        pairs = [(query, chunk.to_index_text()) for chunk in chunks]
        scores: list[float] = model.predict(pairs).tolist()  # type: ignore[union-attr]

        scored = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        logger.debug(
            "Reranked %d chunks → returning top %d (score range %.4f – %.4f)",
            len(chunks),
            min(top_k, len(scored)),
            scored[-1][1] if scored else 0.0,
            scored[0][1] if scored else 0.0,
        )

        return scored[:top_k]

    def is_available(self) -> bool:
        """Return *True* if the cross-encoder model can be loaded."""
        try:
            self._load_model()
            return True
        except Exception:  # noqa: BLE001
            logger.warning(
                "Reranker model '%s' is not available", self._model_name
            )
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker model '%s' …", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model
