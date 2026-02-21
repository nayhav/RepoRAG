"""Evaluation framework for GitRAG retrieval and generation quality."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

from gitrag.config import EvaluationConfig
from gitrag.core.types import CodeChunk


# ---------------------------------------------------------------------------
# Stopwords (lightweight built-in set for faithfulness heuristic)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "no", "nor", "so", "yet", "both", "each", "few", "more", "most",
    "other", "some", "such", "than", "too", "very", "just", "about",
    "above", "after", "again", "all", "also", "any", "because", "before",
    "below", "between", "during", "further", "here", "how", "into", "it",
    "its", "itself", "me", "my", "myself", "once", "only", "our", "ours",
    "out", "over", "own", "same", "she", "he", "her", "him", "his",
    "that", "them", "then", "there", "these", "they", "this", "those",
    "through", "under", "until", "up", "we", "what", "when", "where",
    "which", "while", "who", "whom", "why", "you", "your",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase alpha-numeric tokenisation."""
    import re
    return re.findall(r"[a-z0-9_]+", text.lower())


def _extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks from markdown-style text."""
    import re
    return re.findall(r"```[\s\S]*?```", text)


# ---------------------------------------------------------------------------
# Evaluation Framework
# ---------------------------------------------------------------------------


class EvaluationFramework:
    """Computes retrieval and generation quality metrics for GitRAG."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    # -- Retrieval Metrics ---------------------------------------------------

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """Fraction of top-k retrieved items that are relevant."""
        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0
        relevant_count = sum(1 for rid in top_k if rid in relevant_ids)
        return relevant_count / len(top_k)

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """Fraction of all relevant items that appear in top-k."""
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        found = sum(1 for rid in top_k if rid in relevant_ids)
        return found / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
        """Mean Reciprocal Rank: 1 / rank of first relevant result."""
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int,
    ) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        top_k = retrieved_ids[:k]

        # DCG: sum of 1/log2(rank+1) for each relevant hit
        dcg = 0.0
        for i, rid in enumerate(top_k):
            if rid in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)  # i+2 because rank is 1-indexed

        # Ideal DCG: all relevant items at the top
        ideal_hits = min(len(relevant_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    # -- Generation Metrics --------------------------------------------------

    @staticmethod
    def faithfulness_score(answer: str, context_chunks: list[str]) -> float:
        """Heuristic faithfulness: fraction of non-stopword answer tokens found in context."""
        answer_tokens = _tokenize(answer)
        content_tokens = [t for t in answer_tokens if t not in _STOPWORDS]
        if not content_tokens:
            return 1.0

        context_text = " ".join(context_chunks)
        context_token_set = set(_tokenize(context_text))

        matched = sum(1 for t in content_tokens if t in context_token_set)
        return matched / len(content_tokens)

    @staticmethod
    def citation_coverage(answer: str, citations: list[Any]) -> float:
        """Fraction of answer paragraphs that have at least one citation."""
        paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
        if not paragraphs:
            return 0.0
        if not citations:
            return 0.0

        # Build set of snippets / file references from citations for matching
        citation_markers: set[str] = set()
        for c in citations:
            if hasattr(c, "file_path"):
                citation_markers.add(c.file_path)
            if hasattr(c, "symbol_name") and c.symbol_name:
                citation_markers.add(c.symbol_name)
            if hasattr(c, "snippet") and c.snippet:
                citation_markers.add(c.snippet)
            # Also support plain-string citations
            if isinstance(c, str):
                citation_markers.add(c)

        covered = 0
        for para in paragraphs:
            para_lower = para.lower()
            if any(m.lower() in para_lower for m in citation_markers if m):
                covered += 1
        return covered / len(paragraphs)

    @classmethod
    def hallucination_score(
        cls,
        answer: str,
        context_chunks: list[str],
    ) -> float:
        """1 - faithfulness_score.  Also flags code blocks absent from context."""
        base = 1.0 - cls.faithfulness_score(answer, context_chunks)

        # Penalise code blocks that don't appear in any context chunk
        code_blocks = _extract_code_blocks(answer)
        if code_blocks:
            context_joined = "\n".join(context_chunks)
            unmatched = 0
            for block in code_blocks:
                # Strip fences for comparison
                inner = block.strip("`").strip()
                if inner and inner not in context_joined:
                    unmatched += 1
            block_penalty = unmatched / len(code_blocks)
            # Blend: 70% token-based, 30% code-block-based
            base = 0.7 * base + 0.3 * block_penalty

        return min(base, 1.0)

    # -- Benchmarking --------------------------------------------------------

    @staticmethod
    def measure_latency(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
        """Time a function call.  Returns ``(result, latency_seconds)``."""
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    # -- Synthetic Evaluation ------------------------------------------------

    @staticmethod
    def generate_synthetic_queries(
        chunks: list[CodeChunk],
        n: int = 50,
    ) -> list[dict[str, Any]]:
        """Generate synthetic test queries from code chunks.

        For each selected chunk a question of the form
        *"What does {symbol_name} do?"* is created.  The ground truth is the
        ``chunk_id`` of the source chunk.
        """
        if not chunks:
            return []

        # Prefer chunks with meaningful symbol names
        candidates = [c for c in chunks if c.symbol_name and c.symbol_name != "<module>"]
        if not candidates:
            candidates = list(chunks)

        selected = random.sample(candidates, min(n, len(candidates)))

        _TEMPLATES: dict[str, str] = {
            "explain": "What does {symbol} do?",
            "how_to": "How do I use {symbol}?",
            "search": "Where is {symbol} defined?",
        }

        queries: list[dict[str, Any]] = []
        intent_keys = list(_TEMPLATES.keys())
        for chunk in selected:
            intent = random.choice(intent_keys)
            query_text = _TEMPLATES[intent].format(symbol=chunk.symbol_name)
            queries.append({
                "query": query_text,
                "relevant_chunk_ids": [chunk.chunk_id],
                "intent": intent,
            })
        return queries

    # -- Reporting -----------------------------------------------------------

    def run_evaluation(
        self,
        queries: list[dict[str, Any]],
        retriever: Any,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Run all queries through *retriever* and compute aggregate metrics.

        ``retriever`` must implement ``retrieve(query: str, top_k: int)``
        returning a list of :class:`RetrievalResult`.
        """
        precisions: list[float] = []
        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []

        per_query: list[dict[str, Any]] = []

        for q in queries:
            query_text: str = q["query"]
            relevant_ids: set[str] = set(q["relevant_chunk_ids"])

            results, latency = self.measure_latency(retriever.retrieve, query_text, top_k)
            retrieved_ids = [r.chunk.chunk_id for r in results]

            p = self.precision_at_k(retrieved_ids, relevant_ids, top_k)
            r = self.recall_at_k(retrieved_ids, relevant_ids, top_k)
            m = self.mrr(retrieved_ids, relevant_ids)
            n = self.ndcg_at_k(retrieved_ids, relevant_ids, top_k)

            precisions.append(p)
            recalls.append(r)
            mrrs.append(m)
            ndcgs.append(n)
            latencies.append(latency)

            per_query.append({
                "query": query_text,
                "intent": q.get("intent", ""),
                "precision_at_k": p,
                "recall_at_k": r,
                "mrr": m,
                "ndcg_at_k": n,
                "latency_seconds": latency,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": list(relevant_ids),
            })

        total = len(queries) or 1
        return {
            "num_queries": len(queries),
            "top_k": top_k,
            "avg_precision_at_k": sum(precisions) / total,
            "avg_recall_at_k": sum(recalls) / total,
            "avg_mrr": sum(mrrs) / total,
            "avg_ndcg_at_k": sum(ndcgs) / total,
            "avg_latency_seconds": sum(latencies) / total,
            "per_query": per_query,
        }

    def save_report(self, results: dict[str, Any], output_dir: str) -> None:
        """Save evaluation results as JSON to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "eval_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
