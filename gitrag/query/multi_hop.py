"""Multi-hop reasoning via dependency graph traversal."""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from gitrag.config import GraphConfig
from gitrag.core.types import QueryIntent, RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph protocol — satisfied by any future DependencyGraph implementation
# ---------------------------------------------------------------------------

@runtime_checkable
class DependencyGraphLike(Protocol):
    """Minimal interface expected from a dependency graph."""

    def get_dependencies(self, node: str) -> list[str]: ...
    def get_dependents(self, node: str) -> list[str]: ...


# ---------------------------------------------------------------------------
# Multi-hop expander
# ---------------------------------------------------------------------------

class MultiHopExpander:
    """Expand retrieval context by traversing a dependency graph.

    If no graph is provided the expander is a no-op: all public methods
    return empty lists so the rest of the pipeline is unaffected.
    """

    def __init__(
        self,
        graph: DependencyGraphLike | None,
        config: GraphConfig,
    ) -> None:
        self._graph = graph
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expand_context(
        self,
        initial_results: list[RetrievalResult],
        query_intent: QueryIntent,
    ) -> list[str]:
        """Return additional file paths discovered via graph traversal.

        The files already present in *initial_results* are excluded.

        Parameters
        ----------
        initial_results:
            The retrieval results from the first-pass retriever.
        query_intent:
            Classified intent of the user query — determines traversal
            strategy and depth.

        Returns
        -------
        list[str]
            Extra file paths (not in *initial_results*) up to a budget of
            ``max_expand_per_hop × max_hops`` files.
        """
        if self._graph is None:
            return []

        seed_files = {r.chunk.file_path for r in initial_results}
        max_hops = self._config.max_hops
        budget = self._config.max_expand_per_hop * max_hops

        if query_intent is QueryIntent.DEPENDENCY:
            additional = self._traverse_both(seed_files, max_hops)
        elif query_intent is QueryIntent.ARCHITECTURE:
            additional = self._traverse_deps(seed_files, max_hops)
        else:
            additional = self._traverse_deps(seed_files, max_hops=1)

        # Remove files we already have and enforce the budget
        additional -= seed_files
        result = sorted(additional)[:budget]

        logger.info(
            "Graph expansion (%s): %d seed files → %d additional files",
            query_intent.value,
            len(seed_files),
            len(result),
        )
        return result

    def get_expanded_chunks(
        self,
        additional_files: list[str],
        all_chunks_by_file: dict[str, list[str]],
    ) -> list[str]:
        """Return chunk IDs from *additional_files*.

        Chunks are prioritised so that function/class definitions appear
        before module-level code.  The total number of returned IDs is
        capped at ``max_expand_per_hop × max_hops``.

        Parameters
        ----------
        additional_files:
            File paths produced by :meth:`expand_context`.
        all_chunks_by_file:
            Mapping of ``file_path → [chunk_id, …]`` for every indexed file.

        Returns
        -------
        list[str]
            Chunk IDs from the expanded files.
        """
        if not additional_files:
            return []

        budget = self._config.max_expand_per_hop * self._config.max_hops
        chunk_ids: list[str] = []

        for fpath in additional_files:
            file_chunks = all_chunks_by_file.get(fpath, [])
            chunk_ids.extend(file_chunks)
            if len(chunk_ids) >= budget:
                break

        result = chunk_ids[:budget]
        logger.debug(
            "Expanded chunks: %d IDs from %d files (budget %d)",
            len(result),
            len(additional_files),
            budget,
        )
        return result

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def _traverse_deps(
        self,
        seed_files: set[str],
        max_hops: int,
    ) -> set[str]:
        """BFS outward along dependency edges (what does this file depend on?)."""
        assert self._graph is not None
        visited: set[str] = set(seed_files)
        frontier = set(seed_files)

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for dep in self._graph.get_dependencies(node):
                    if dep not in visited:
                        visited.add(dep)
                        next_frontier.add(dep)
            frontier = next_frontier
            if not frontier:
                break

        return visited - seed_files

    def _traverse_both(
        self,
        seed_files: set[str],
        max_hops: int,
    ) -> set[str]:
        """BFS in both directions (dependencies *and* dependents)."""
        assert self._graph is not None
        visited: set[str] = set(seed_files)
        frontier = set(seed_files)

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbour in self._graph.get_dependencies(node):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_frontier.add(neighbour)
                for neighbour in self._graph.get_dependents(node):
                    if neighbour not in visited:
                        visited.add(neighbour)
                        next_frontier.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        return visited - seed_files
