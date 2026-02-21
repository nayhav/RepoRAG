"""Core data types for GitRAG.

Every module in the system communicates through these types.
They are intentionally simple dataclasses — no ORM, no heavy base classes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Language & File Types
# ---------------------------------------------------------------------------

class Language(str, Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    CPP = "cpp"
    JAVA = "java"
    C = "c"
    GO = "go"
    RUST = "rust"
    MARKDOWN = "markdown"
    RST = "rst"
    TEXT = "text"
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    UNKNOWN = "unknown"


# Languages that support AST chunking via tree-sitter
AST_SUPPORTED_LANGUAGES = frozenset({
    Language.PYTHON,
    Language.JAVASCRIPT,
    Language.TYPESCRIPT,
    Language.CPP,
    Language.JAVA,
    Language.C,
    Language.GO,
    Language.RUST,
})

# Languages treated as documentation
DOC_LANGUAGES = frozenset({
    Language.MARKDOWN,
    Language.RST,
    Language.TEXT,
})


# ---------------------------------------------------------------------------
# Symbol Kinds (AST-level)
# ---------------------------------------------------------------------------

class SymbolKind(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    NAMESPACE = "namespace"
    VARIABLE = "variable"
    CONSTANT = "constant"
    IMPORT = "import"
    DOCUMENTATION = "documentation"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Ingested File
# ---------------------------------------------------------------------------

@dataclass
class IngestedFile:
    """Raw file loaded from the repository."""

    path: Path                    # Relative path from repo root
    abs_path: Path                # Absolute path on disk
    content: str                  # Full file content
    language: Language
    size_bytes: int
    encoding: str = "utf-8"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Code Chunk — the atomic unit of indexing
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A semantically meaningful piece of code or documentation.

    This is the fundamental retrieval unit.  Each chunk carries enough metadata
    to reconstruct its location, provenance, and structural context.
    """

    chunk_id: str                      # Deterministic ID (hash of path + range)
    file_path: str                     # Relative path from repo root
    language: Language
    symbol_name: str                   # e.g. "MyClass.my_method"
    symbol_kind: SymbolKind
    content: str                       # The actual code/text
    start_line: int                    # 1-indexed
    end_line: int                      # 1-indexed, inclusive
    # Contextual metadata
    docstring: str = ""                # Extracted docstring/comment
    imports: list[str] = field(default_factory=list)     # Import statements in scope
    references: list[str] = field(default_factory=list)  # Outgoing symbol references
    parent_symbol: str = ""            # Enclosing class/namespace
    # For retrieval metadata filtering
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def location(self) -> str:
        """Human-readable location string for citations."""
        return f"{self.file_path}:{self.start_line}-{self.end_line}"

    def to_index_text(self) -> str:
        """Text representation used for embedding and BM25 indexing.

        Includes structural context (file path, symbol name, docstring)
        so the embedding captures semantic meaning beyond raw code.
        """
        parts = [
            f"File: {self.file_path}",
            f"Symbol: {self.symbol_name} ({self.symbol_kind.value})",
        ]
        if self.docstring:
            parts.append(f"Documentation: {self.docstring}")
        if self.imports:
            parts.append(f"Imports: {', '.join(self.imports[:10])}")
        parts.append(f"Code:\n{self.content}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Dependency Edge (for the graph)
# ---------------------------------------------------------------------------

@dataclass
class DependencyEdge:
    """Directed edge in the dependency graph."""

    source_file: str          # File that imports/references
    target_file: str          # File being imported/referenced
    source_symbol: str = ""   # Symbol doing the importing
    target_symbol: str = ""   # Symbol being imported
    edge_type: str = "import" # import, call, inheritance, etc.


# ---------------------------------------------------------------------------
# Query & Intent
# ---------------------------------------------------------------------------

class QueryIntent(str, Enum):
    EXPLAIN = "explain"             # Explain a function/class/module
    BUG_FINDING = "bug_finding"     # Find potential bugs
    ARCHITECTURE = "architecture"   # Architecture/design question
    REFACTOR = "refactor"           # Refactoring suggestion
    DEPENDENCY = "dependency"       # Dependency tracing
    SEARCH = "search"               # General code search
    HOW_TO = "how_to"               # How to use / how does X work
    UNKNOWN = "unknown"


@dataclass
class ParsedQuery:
    """User query after intent classification and reformulation."""

    original: str
    reformulated: str
    intent: QueryIntent
    # Extracted entities (file names, function names, etc.)
    entities: list[str] = field(default_factory=list)
    # If this is a follow-up, the context from prior turns
    context_from_history: str = ""
    # Retrieval depth override based on intent
    retrieval_depth: int = 10


# ---------------------------------------------------------------------------
# Retrieval Result
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """A single retrieved chunk with relevance scores."""

    chunk: CodeChunk
    # Scores from different retrievers (higher = more relevant)
    vector_score: float = 0.0
    bm25_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    # Final score used for ordering
    final_score: float = 0.0
    # Provenance
    retrieval_method: str = ""  # "vector", "bm25", "hybrid", "graph_expand"


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in a multi-turn conversation."""

    role: str        # "user" or "assistant"
    content: str
    # Chunks that were retrieved for this turn (assistant turns only)
    cited_chunks: list[str] = field(default_factory=list)  # chunk_ids


@dataclass
class Conversation:
    """Full conversation state."""

    conversation_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    summary: str = ""  # Rolling summary for long conversations
    repo_path: str = ""

    def add_turn(self, role: str, content: str, cited_chunks: list[str] | None = None) -> None:
        self.turns.append(ConversationTurn(
            role=role,
            content=content,
            cited_chunks=cited_chunks or [],
        ))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class GeneratedAnswer:
    """Response from the LLM with citation metadata."""

    content: str
    citations: list[Citation] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Citation:
    """A reference back to source code."""

    file_path: str
    start_line: int
    end_line: int
    symbol_name: str = ""
    chunk_id: str = ""
    snippet: str = ""  # Short excerpt for display
