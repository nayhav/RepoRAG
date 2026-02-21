"""Abstract base class for embedding models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """Interface that all embedding backends must implement."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of document texts.

        Returns an array of shape ``(N, dimensions)``.
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns an array of shape ``(dimensions,)``.
        """
        ...
