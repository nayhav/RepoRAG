"""Hybrid retriever combining dense vector search, sparse BM25, RRF, and reranking."""

from __future__ import annotations

import logging
from typing import Any

from gitrag.config import RetrievalConfig
from gitrag.core.types import CodeChunk, RetrievalResult
from gitrag.retrieval.fusion import normalize_scores, reciprocal_rank_fusion
from gitrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Orchestrates the full hybrid retrieval pipeline.

    Pipeline stages:
    1. Embed the query.
    2. Retrieve candidates from vector store and BM25 index.
    3. Fuse results via Reciprocal Rank Fusion.
    4. (Optional) Rerank the top fusion candidates with a cross-encoder.
    5. Return :class:`RetrievalResult` objects sorted by ``final_score``.
    """

    def __init__(
        self,
        config: RetrievalConfig,
        vector_store: Any,
        bm25_store: Any,
        embedder: Any,
        reranker: Reranker | None = None,
    ) -> None:
        self._config = config
        self._vector_store = vector_store
        self._bm25_store = bm25_store
        self._embedder = embedder
        self._reranker = reranker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Run the full hybrid retrieval pipeline.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Override for the final number of results.  Falls back to
            ``rerank_top_k`` (if reranking) or ``fusion_top_k`` from config.
        filters:
            Optional metadata filters forwarded to the vector store.

        Returns
        -------
        list[RetrievalResult]
            Results sorted by ``final_score`` descending.
        """
        cfg = self._config

        # 1. Embed the query -----------------------------------------------
        query_embedding = self._embedder.embed_query(query)

        # 2. Retrieve from both stores ------------------------------------
        vector_results: list[tuple[str, float]] = self._vector_store.query(
            query_embedding, top_k=cfg.vector_top_k, filters=filters,
        )
        bm25_results: list[tuple[str, float]] = self._bm25_store.query(
            query, top_k=cfg.bm25_top_k,
        )

        vector_ids = {cid for cid, _ in vector_results}
        bm25_ids = {cid for cid, _ in bm25_results}
        overlap = vector_ids & bm25_ids

        logger.info(
            "Vector: %d hits | BM25: %d hits | overlap: %d",
            len(vector_results),
            len(bm25_results),
            len(overlap),
        )

        # 3. Reciprocal Rank Fusion ----------------------------------------
        fused = reciprocal_rank_fusion(
            [vector_results, bm25_results], k=cfg.rrf_k,
        )
        fused = fused[: cfg.fusion_top_k]

        # Build lookup maps for per-retriever scores
        vector_score_map = dict(vector_results)
        bm25_score_map = dict(bm25_results)
        fused_score_map = dict(fused)

        # Collect all chunk IDs we need
        candidate_ids = [cid for cid, _ in fused]
        chunks_map = self._fetch_chunks(candidate_ids)

        # Remove IDs that could not be resolved
        candidate_ids = [cid for cid in candidate_ids if cid in chunks_map]
        if not candidate_ids:
            logger.warning("No chunks could be resolved — returning empty.")
            return []

        # 4. Optional reranking -------------------------------------------
        rerank_score_map: dict[str, float] = {}
        final_ids: list[str]

        if self._reranker is not None and cfg.enable_reranking:
            rerank_candidates = [chunks_map[cid] for cid in candidate_ids]
            effective_top_k = top_k if top_k is not None else cfg.rerank_top_k
            reranked = self._reranker.rerank(
                query, rerank_candidates, top_k=effective_top_k,
            )
            rerank_score_map = {
                chunk.chunk_id: score for chunk, score in reranked
            }
            final_ids = [chunk.chunk_id for chunk, _ in reranked]
            logger.info("Reranked → %d results", len(final_ids))
        else:
            effective_top_k = top_k if top_k is not None else cfg.fusion_top_k
            final_ids = candidate_ids[:effective_top_k]

        # 5. Build RetrievalResult objects ---------------------------------
        scores: dict[str, dict[str, float]] = {}
        for cid in final_ids:
            entry: dict[str, float] = {
                "vector_score": vector_score_map.get(cid, 0.0),
                "bm25_score": bm25_score_map.get(cid, 0.0),
                "fused_score": fused_score_map.get(cid, 0.0),
                "rerank_score": rerank_score_map.get(cid, 0.0),
            }
            # Final score is the rerank score when available, else fused.
            entry["final_score"] = (
                entry["rerank_score"]
                if rerank_score_map
                else entry["fused_score"]
            )
            scores[cid] = entry

        results = self._build_retrieval_results(final_ids, scores, chunks_map)
        results.sort(key=lambda r: r.final_score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_retrieval_results(
        self,
        chunk_ids: list[str],
        scores: dict[str, dict[str, float]],
        chunks_map: dict[str, CodeChunk],
    ) -> list[RetrievalResult]:
        """Construct :class:`RetrievalResult` objects from scored chunk IDs."""
        results: list[RetrievalResult] = []
        for cid in chunk_ids:
            chunk = chunks_map.get(cid)
            if chunk is None:
                continue
            s = scores.get(cid, {})
            method_parts: list[str] = []
            if s.get("vector_score", 0.0) > 0.0:
                method_parts.append("vector")
            if s.get("bm25_score", 0.0) > 0.0:
                method_parts.append("bm25")
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    vector_score=s.get("vector_score", 0.0),
                    bm25_score=s.get("bm25_score", 0.0),
                    fused_score=s.get("fused_score", 0.0),
                    rerank_score=s.get("rerank_score", 0.0),
                    final_score=s.get("final_score", 0.0),
                    retrieval_method="hybrid" if len(method_parts) > 1 else (
                        method_parts[0] if method_parts else "hybrid"
                    ),
                ),
            )
        return results

    def _fetch_chunks(self, chunk_ids: list[str]) -> dict[str, CodeChunk]:
        """Retrieve :class:`CodeChunk` objects by ID from the vector store."""
        chunks_map: dict[str, CodeChunk] = {}
        for cid in chunk_ids:
            chunk = self._vector_store.get_chunk(cid)
            if chunk is not None:
                chunks_map[cid] = chunk
            else:
                logger.debug("Chunk '%s' not found in vector store", cid)
        return chunks_map
