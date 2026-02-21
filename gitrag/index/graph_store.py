"""Dependency graph built from code chunk import information."""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path, PurePosixPath
from typing import Any

import networkx as nx

from gitrag.core.types import CodeChunk

logger = logging.getLogger(__name__)


class DependencyGraph:
    """Directed graph of file-level import/dependency relationships."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_from_chunks(
        self,
        chunks: list[CodeChunk],
        all_files: list[str] | None = None,
    ) -> None:
        """Populate the graph from chunk import lists.

        Parameters
        ----------
        chunks:
            Parsed code chunks whose ``.imports`` and ``.file_path`` are used.
        all_files:
            Every file path in the repository (relative to repo root).
            Used for import resolution.  When *None*, the set is derived
            from the chunks themselves.
        """
        if all_files is None:
            all_files = sorted({c.file_path for c in chunks})

        for chunk in chunks:
            src = chunk.file_path
            if not self._graph.has_node(src):
                self._graph.add_node(src)

            for imp in chunk.imports:
                resolved = self.resolve_import(imp, src, all_files)
                if resolved is not None and resolved != src:
                    self.add_edge(src, resolved, edge_type="import")

        logger.info(
            "Dependency graph built – %d nodes, %d edges",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: str = "import",
    ) -> None:
        """Add a directed edge *source* → *target*."""
        self._graph.add_edge(source, target, edge_type=edge_type)

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_dependencies(self, file_path: str, depth: int = 1) -> set[str]:
        """Files that *file_path* depends on (outgoing edges), up to *depth* hops."""
        return self._bfs(file_path, depth, reverse=False)

    def get_dependents(self, file_path: str, depth: int = 1) -> set[str]:
        """Files that depend on *file_path* (incoming edges), up to *depth* hops."""
        return self._bfs(file_path, depth, reverse=True)

    def get_related_files(
        self,
        file_path: str,
        max_hops: int,
        max_nodes: int,
    ) -> list[str]:
        """Combined dependencies and dependents, sorted by proximity.

        Returns at most *max_nodes* file paths (excluding *file_path* itself).
        """
        visited: dict[str, int] = {}  # file -> min_hop
        queue: deque[tuple[str, int]] = deque([(file_path, 0)])

        while queue:
            node, hop = queue.popleft()
            if hop > max_hops:
                continue
            if node in visited:
                continue
            visited[node] = hop

            for neighbour in self._graph.successors(node):
                if neighbour not in visited:
                    queue.append((neighbour, hop + 1))
            for neighbour in self._graph.predecessors(node):
                if neighbour not in visited:
                    queue.append((neighbour, hop + 1))

        visited.pop(file_path, None)

        sorted_files = sorted(visited, key=lambda f: visited[f])
        return sorted_files[:max_nodes]

    # ------------------------------------------------------------------
    # Import resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_import(
        import_str: str,
        source_file: str,
        all_files: list[str],
    ) -> str | None:
        """Best-effort resolution of an import string to a repo file path.

        Handles common conventions for Python, JS/TS, Java, and C/C++.
        """
        stripped = import_str.strip()
        all_set = set(all_files)

        # ---- Python -------------------------------------------------
        # "from foo.bar import baz"  /  "import foo.bar"
        if stripped.startswith("from ") or stripped.startswith("import "):
            module = stripped.replace("from ", "").replace("import ", "").split()[0]
            parts = module.split(".")
            candidates = [
                "/".join(parts) + ".py",
                "/".join(parts) + "/__init__.py",
            ]
            for c in candidates:
                if c in all_set:
                    return c

        # ---- JS / TS relative imports --------------------------------
        # "./utils" / "../helpers/foo"
        if stripped.startswith("."):
            source_dir = str(PurePosixPath(source_file).parent)
            base = str(PurePosixPath(source_dir) / stripped)
            # Normalise ../ etc.
            base = str(PurePosixPath(base))
            extensions = ["", ".js", ".ts", ".jsx", ".tsx"]
            index_files = ["/index.js", "/index.ts", "/index.jsx", "/index.tsx"]
            for ext in extensions:
                if (base + ext) in all_set:
                    return base + ext
            for idx in index_files:
                if (base + idx) in all_set:
                    return base + idx

        # ---- Java ----------------------------------------------------
        # "import com.foo.Bar" -> com/foo/Bar.java
        if stripped.startswith("import ") and "." in stripped:
            parts = stripped.split()[-1].rstrip(";").split(".")
            candidate = "/".join(parts) + ".java"
            if candidate in all_set:
                return candidate

        # ---- C / C++ includes ----------------------------------------
        # #include "foo/bar.h"
        if stripped.startswith("#include"):
            match = stripped.split('"')
            if len(match) >= 2:
                header = match[1]
                if header in all_set:
                    return header

        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Serialize the graph to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._graph)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Dependency graph saved to %s", path)

    def load(self, path: Path) -> None:
        """Load a previously saved graph."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._graph = nx.node_link_graph(data, directed=True)
        logger.info(
            "Dependency graph loaded from %s (%d nodes, %d edges)",
            path,
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Summary statistics for the graph."""
        g = self._graph
        return {
            "node_count": g.number_of_nodes(),
            "edge_count": g.number_of_edges(),
            "connected_components": nx.number_weakly_connected_components(g),
            "density": nx.density(g),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _bfs(self, start: str, depth: int, *, reverse: bool) -> set[str]:
        """BFS up to *depth* hops along successors or predecessors."""
        if start not in self._graph:
            return set()

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start, 0)])

        while queue:
            node, hop = queue.popleft()
            if hop > depth:
                continue
            if node in visited:
                continue
            visited.add(node)

            neighbours = (
                self._graph.predecessors(node) if reverse else self._graph.successors(node)
            )
            for nb in neighbours:
                if nb not in visited:
                    queue.append((nb, hop + 1))

        visited.discard(start)
        return visited
