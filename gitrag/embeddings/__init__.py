"""Embedding generation for code chunks."""

from gitrag.embeddings.base import BaseEmbedder
from gitrag.embeddings.local import LocalEmbedder

__all__ = ["BaseEmbedder", "LocalEmbedder"]
