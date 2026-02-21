"""Abstract base class for chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from gitrag.core.types import CodeChunk, IngestedFile


class BaseChunker(ABC):
    """Base interface that all chunkers implement."""

    @abstractmethod
    def chunk(self, file: IngestedFile) -> list[CodeChunk]:
        """Split a single ingested file into semantically meaningful chunks."""
        ...
