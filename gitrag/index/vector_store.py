"""ChromaDB-based vector store for code chunk embeddings."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
import numpy as np

from gitrag.config import IndexConfig
from gitrag.core.types import CodeChunk

logger = logging.getLogger(__name__)

_CHROMA_BATCH_LIMIT = 5000


class VectorStore:
    """Persistent vector index backed by ChromaDB with cosine similarity."""

    def __init__(self, config: IndexConfig) -> None:
        self._config = config
        self._client = chromadb.PersistentClient(path=config.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore initialised – collection=%s, count=%d",
            config.collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[CodeChunk], embeddings: np.ndarray) -> None:
        """Upsert chunks with pre-computed embeddings (batched)."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) length mismatch"
            )

        ids = [c.chunk_id for c in chunks]
        documents = [c.to_index_text() for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "language": c.language.value,
                "symbol_name": c.symbol_name,
                "symbol_kind": c.symbol_kind.value,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in chunks
        ]
        emb_list = embeddings.tolist()

        for start in range(0, len(ids), _CHROMA_BATCH_LIMIT):
            end = start + _CHROMA_BATCH_LIMIT
            self._collection.upsert(
                ids=ids[start:end],
                embeddings=emb_list[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info("Upserted %d chunks into vector store", len(ids))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        embedding: np.ndarray,
        top_k: int,
        where: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        """Return ``(chunk_id, similarity_score)`` tuples sorted by relevance."""
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding.tolist()],
            "n_results": top_k,
        }
        effective_where = where or filters
        if effective_where:
            kwargs["where"] = effective_where

        results = self._collection.query(**kwargs)

        ids: list[str] = results["ids"][0]  # type: ignore[index]
        distances: list[float] = results["distances"][0]  # type: ignore[index]

        return [(cid, 1.0 - dist) for cid, dist in zip(ids, distances)]

    def get_chunk(self, chunk_id: str) -> CodeChunk | None:
        """Retrieve a single :class:`CodeChunk` by id, or *None*."""
        result = self._collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        doc = result["documents"][0]  # type: ignore[index]
        meta = result["metadatas"][0]  # type: ignore[index]
        from gitrag.core.types import Language, SymbolKind
        return CodeChunk(
            chunk_id=chunk_id,
            file_path=meta.get("file_path", ""),
            language=Language(meta.get("language", "unknown")),
            symbol_name=meta.get("symbol_name", ""),
            symbol_kind=SymbolKind(meta.get("symbol_kind", "unknown")),
            content=doc or "",
            start_line=meta.get("start_line", 0),
            end_line=meta.get("end_line", 0),
        )

    def get_chunks_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Retrieve documents and metadata keyed by chunk id."""
        if not ids:
            return {}

        results = self._collection.get(ids=ids, include=["documents", "metadatas"])

        out: dict[str, dict[str, Any]] = {}
        for cid, doc, meta in zip(
            results["ids"],
            results["documents"],  # type: ignore[arg-type]
            results["metadatas"],  # type: ignore[arg-type]
        ):
            out[cid] = {"document": doc, "metadata": meta}
        return out

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all data from the collection."""
        self._client.delete_collection(self._config.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store cleared")

    def count(self) -> int:
        """Number of indexed chunks."""
        return self._collection.count()
