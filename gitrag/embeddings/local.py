"""Local embedding model backed by sentence-transformers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from gitrag.config import EmbeddingsConfig
from gitrag.embeddings.base import BaseEmbedder

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class LocalEmbedder(BaseEmbedder):
    """Generate embeddings using a locally-running sentence-transformers model.

    The model is lazily loaded on first use so that importing this module
    stays cheap.
    """

    def __init__(self, config: EmbeddingsConfig) -> None:
        self._config = config
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # BaseEmbedder interface
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document texts.

        Returns an array of shape ``(N, dimensions)``.
        """
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)

        model = self._load_model()

        show_progress = len(texts) > self._config.batch_size
        if show_progress:
            logger.info(
                "Embedding %d texts in batches of %d …",
                len(texts),
                self._config.batch_size,
            )

        embeddings: np.ndarray = model.encode(
            texts,
            batch_size=self._config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self._config.normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns an array of shape ``(dimensions,)``.
        """
        model = self._load_model()
        text = f"{self._config.query_prefix}{query}" if self._config.query_prefix else query

        embedding: np.ndarray = model.encode(
            text,
            normalize_embeddings=self._config.normalize,
            convert_to_numpy=True,
        )
        return embedding

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Force the model to load (useful for startup health-checks)."""
        self._load_model()
        logger.info("Embedder warmed up – model ready on %s", self._resolve_device())

    def _load_model(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        device = self._resolve_device()
        logger.info(
            "Loading embedding model %s on %s …",
            self._config.model_name,
            device,
        )
        self._model = SentenceTransformer(self._config.model_name, device=device)
        return self._model

    def _resolve_device(self) -> str:
        if self._config.device:
            return self._config.device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
