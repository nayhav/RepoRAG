"""AST-aware chunker using tree-sitter.

Parses source files into an AST and extracts semantically meaningful chunks
(functions, classes, methods, imports, and orphan code).
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from gitrag.config import ChunkingConfig
from gitrag.core.types import (
    CodeChunk,
    IngestedFile,
    Language,
    SymbolKind,
)

from .base import BaseChunker
from .text_chunker import TextChunker

if TYPE_CHECKING:
    from tree_sitter import Node, Parser, Tree

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _chunk_id(file_path: str, start_line: int, end_line: int) -> str:
    raw = f"{file_path}:{start_line}:{end_line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Per-language configuration
# ---------------------------------------------------------------------------

# Node types that represent top-level definitions we want to extract.
_DEFINITION_TYPES: dict[Language, set[str]] = {
    Language.PYTHON: {
        "function_definition",
        "class_definition",
        "decorated_definition",
    },
    Language.JAVASCRIPT: {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
    },
    Language.TYPESCRIPT: {
        "function_declaration",
        "class_declaration",
        "export_statement",
        "lexical_declaration",
    },
    Language.CPP: {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    },
    Language.C: {
        "function_definition",
    },
    Language.JAVA: {
        "class_declaration",
        "method_declaration",
        "interface_declaration",
    },
}

# Node types treated as imports.
_IMPORT_TYPES: dict[Language, set[str]] = {
    Language.PYTHON: {"import_statement", "import_from_statement"},
    Language.JAVASCRIPT: {"import_statement"},
    Language.TYPESCRIPT: {"import_statement"},
    Language.CPP: {"preproc_include"},
    Language.C: {"preproc_include"},
    Language.JAVA: {"import_declaration"},
}

# Mapping from node type to SymbolKind.
_KIND_MAP: dict[str, SymbolKind] = {
    # Python
    "function_definition": SymbolKind.FUNCTION,
    "class_definition": SymbolKind.CLASS,
    # JS / TS
    "function_declaration": SymbolKind.FUNCTION,
    "class_declaration": SymbolKind.CLASS,
    "method_definition": SymbolKind.METHOD,
    "arrow_function": SymbolKind.FUNCTION,
    # C / C++
    "class_specifier": SymbolKind.CLASS,
    "struct_specifier": SymbolKind.STRUCT,
    "namespace_definition": SymbolKind.NAMESPACE,
    # Java
    "method_declaration": SymbolKind.METHOD,
    "interface_declaration": SymbolKind.INTERFACE,
}

# Inner node types within classes that represent methods.
_METHOD_TYPES: dict[Language, set[str]] = {
    Language.PYTHON: {"function_definition", "decorated_definition"},
    Language.JAVASCRIPT: {"method_definition"},
    Language.TYPESCRIPT: {"method_definition"},
    Language.CPP: {"function_definition"},
    Language.JAVA: {"method_declaration"},
}


# ---------------------------------------------------------------------------
# Parser factory
# ---------------------------------------------------------------------------

def _create_parser(language: Language) -> Parser | None:
    """Create a tree-sitter Parser for the given language.

    Returns ``None`` if the language grammar is not available.
    """
    try:
        from tree_sitter import Language as TSLanguage, Parser

        if language == Language.PYTHON:
            import tree_sitter_python as tsp
            ts_lang = TSLanguage(tsp.language())
        elif language == Language.JAVASCRIPT:
            import tree_sitter_javascript as tsjs
            ts_lang = TSLanguage(tsjs.language())
        elif language == Language.TYPESCRIPT:
            import tree_sitter_typescript as tsts
            ts_lang = TSLanguage(tsts.language_typescript())
        elif language in (Language.CPP, Language.C):
            import tree_sitter_cpp as tscpp
            ts_lang = TSLanguage(tscpp.language())
        elif language == Language.JAVA:
            import tree_sitter_java as tsjava
            ts_lang = TSLanguage(tsjava.language())
        else:
            log.debug("No tree-sitter grammar for %s", language.value)
            return None

        return Parser(ts_lang)
    except Exception:
        log.warning("Failed to initialise tree-sitter for %s", language.value, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# AST Chunker
# ---------------------------------------------------------------------------

class ASTChunker(BaseChunker):
    """Parse source files with tree-sitter and extract semantic chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._config = config or ChunkingConfig()
        self._parsers: dict[Language, Parser | None] = {}
        self._fallback = TextChunker(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, file: IngestedFile) -> list[CodeChunk]:
        parser = self._get_parser(file.language)
        if parser is None:
            return self._fallback.chunk(file)

        try:
            tree = parser.parse(file.content.encode())
        except Exception:
            log.warning("tree-sitter parse failed for %s, falling back to text chunker",
                        file.path, exc_info=True)
            return self._fallback.chunk(file)

        return self._extract_chunks(file, tree)

    # ------------------------------------------------------------------
    # Parser cache
    # ------------------------------------------------------------------

    def _get_parser(self, language: Language) -> Parser | None:
        if language not in self._parsers:
            self._parsers[language] = _create_parser(language)
        return self._parsers[language]

    # ------------------------------------------------------------------
    # Chunk extraction
    # ------------------------------------------------------------------

    def _extract_chunks(self, file: IngestedFile, tree: Tree) -> list[CodeChunk]:
        root = tree.root_node
        file_path = str(file.path)
        lang = file.language

        # 1. Collect file-level imports
        import_texts = self._collect_imports(root, lang)

        # 2. Walk top-level children and extract definitions
        definition_nodes: list[tuple[Node, SymbolKind, str, str]] = []
        covered_ranges: list[tuple[int, int]] = []  # (start_byte, end_byte)

        import_types = _IMPORT_TYPES.get(lang, set())
        definition_types = _DEFINITION_TYPES.get(lang, set())

        for child in root.children:
            if child.type in import_types:
                covered_ranges.append((child.start_byte, child.end_byte))
                continue

            if child.type == "comment":
                # Comments before definitions will be picked up as docstrings
                continue

            if child.type in definition_types:
                extracted = self._extract_definition(child, lang)
                if extracted:
                    definition_nodes.extend(extracted)
                    covered_ranges.append((child.start_byte, child.end_byte))

        # 3. Build chunks from definitions
        chunks: list[CodeChunk] = []
        for node, kind, name, docstring in definition_nodes:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            content = node.text.decode() if node.text else ""

            # Extract references (identifiers matching imported names)
            references = self._extract_references(node, import_texts) if import_texts else []

            chunk = CodeChunk(
                chunk_id=_chunk_id(file_path, start_line, end_line),
                file_path=file_path,
                language=lang,
                symbol_name=name,
                symbol_kind=kind,
                content=content,
                start_line=start_line,
                end_line=end_line,
                docstring=docstring,
                imports=import_texts if self._config.include_imports else [],
                references=references,
            )

            if _estimate_tokens(content) > self._config.max_chunk_tokens:
                chunks.extend(self._split_large_chunk(chunk, node, lang, import_texts))
            else:
                chunks.append(chunk)

        # 4. Collect orphan code (code between definitions)
        orphan_chunks = self._collect_orphans(file, root, covered_ranges, import_texts)
        chunks.extend(orphan_chunks)

        # 5. If nothing was extracted fall back to text chunker
        if not chunks:
            return self._fallback.chunk(file)

        log.debug("ASTChunker produced %d chunks for %s", len(chunks), file_path)
        return chunks

    # ------------------------------------------------------------------
    # Import collection
    # ------------------------------------------------------------------

    def _collect_imports(self, root: Node, lang: Language) -> list[str]:
        import_types = _IMPORT_TYPES.get(lang, set())
        imports: list[str] = []
        for child in root.children:
            if child.type in import_types:
                text = child.text.decode().strip() if child.text else ""
                if text:
                    imports.append(text)
        return imports

    # ------------------------------------------------------------------
    # Definition extraction
    # ------------------------------------------------------------------

    def _extract_definition(
        self, node: Node, lang: Language,
    ) -> list[tuple[Node, SymbolKind, str, str]]:
        """Extract one or more (node, kind, name, docstring) tuples from a definition node."""
        results: list[tuple[Node, SymbolKind, str, str]] = []

        # Handle decorated definitions (Python)
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in ("function_definition", "class_definition"):
                    kind = _KIND_MAP.get(child.type, SymbolKind.UNKNOWN)
                    name = self._get_node_name(child)
                    docstring = self._get_docstring(child, lang)
                    # Use the full decorated node as the source range
                    results.append((node, kind, name, docstring))
                    break
            return results

        # Handle export_statement wrapping a definition (JS/TS)
        if node.type == "export_statement":
            for child in node.children:
                inner = self._extract_definition(child, lang)
                if inner:
                    # Use the export node as range so `export` keyword is included
                    for _n, kind, name, docstring in inner:
                        results.append((node, kind, name, docstring))
                    return results
            return results

        # Handle lexical_declaration with arrow functions (JS/TS: const foo = () => {})
        if node.type == "lexical_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    value = child.child_by_field_name("value")
                    if value and value.type == "arrow_function":
                        name = self._get_node_name(child)
                        docstring = self._get_docstring(node, lang)
                        results.append((node, SymbolKind.FUNCTION, name, docstring))
                        return results
            return results

        # Standard definition
        kind = _KIND_MAP.get(node.type, SymbolKind.UNKNOWN)
        if kind == SymbolKind.UNKNOWN:
            return results

        name = self._get_node_name(node)
        docstring = self._get_docstring(node, lang)
        results.append((node, kind, name, docstring))

        # For classes, also extract methods as separate definitions
        if kind == SymbolKind.CLASS:
            method_types = _METHOD_TYPES.get(lang, set())
            methods = self._find_methods(node, method_types, name, lang)
            results.extend(methods)

        return results

    def _find_methods(
        self,
        class_node: Node,
        method_types: set[str],
        class_name: str,
        lang: Language,
    ) -> list[tuple[Node, SymbolKind, str, str]]:
        """Recursively find method definitions inside a class body."""
        results: list[tuple[Node, SymbolKind, str, str]] = []
        body = class_node.child_by_field_name("body")
        if body is None:
            # Try finding the block/declaration_list child
            for child in class_node.children:
                if child.type in ("block", "class_body", "declaration_list", "field_declaration_list"):
                    body = child
                    break
        if body is None:
            return results

        for child in body.children:
            actual = child
            # Unwrap decorated_definition in Python
            if child.type == "decorated_definition":
                for inner in child.children:
                    if inner.type in method_types:
                        actual = inner
                        break
                else:
                    continue

            if actual.type in method_types:
                name = f"{class_name}.{self._get_node_name(actual)}"
                docstring = self._get_docstring(actual, lang)
                results.append((child, SymbolKind.METHOD, name, docstring))

        return results

    # ------------------------------------------------------------------
    # Name / docstring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_node_name(node: Node) -> str:
        """Extract the symbol name from a node."""
        # Try the 'name' field first (works for most definition types)
        name_node = node.child_by_field_name("name")
        if name_node and name_node.text:
            return name_node.text.decode()

        # For variable_declarator (JS arrow fns)
        name_node = node.child_by_field_name("name")
        if name_node and name_node.text:
            return name_node.text.decode()

        # Fallback: first identifier child
        for child in node.children:
            if child.type == "identifier" and child.text:
                return child.text.decode()
            if child.type == "type_identifier" and child.text:
                return child.text.decode()

        return "<anonymous>"

    @staticmethod
    def _get_docstring(node: Node, lang: Language) -> str:
        """Extract the docstring or leading comment for a definition."""
        # Python: first expression_statement with a string inside the body
        if lang == Language.PYTHON:
            body = node.child_by_field_name("body")
            if body and body.children:
                first = body.children[0]
                if first.type == "expression_statement" and first.children:
                    inner = first.children[0]
                    if inner.type == "string" and inner.text:
                        return inner.text.decode().strip("\"'")

        # Check for a comment node immediately preceding this node
        prev = node.prev_named_sibling
        if prev and prev.type == "comment" and prev.text:
            return prev.text.decode().lstrip("/#* ").strip()

        return ""

    # ------------------------------------------------------------------
    # Reference extraction
    # ------------------------------------------------------------------

    def _extract_references(self, node: Node, import_texts: list[str]) -> list[str]:
        """Find identifier nodes that match imported names."""
        # Build a set of imported symbol names
        imported_names: set[str] = set()
        for imp in import_texts:
            # Extract last token of each import line as the symbol name
            parts = imp.split()
            if parts:
                imported_names.add(parts[-1].strip(",;"))
            # Also handle "from x import y, z"
            if "import" in imp:
                after_import = imp.split("import", 1)[-1]
                for name in after_import.split(","):
                    name = name.strip().split(" as ")[-1].strip()
                    if name:
                        imported_names.add(name)

        if not imported_names:
            return []

        found: set[str] = set()
        self._walk_identifiers(node, imported_names, found)
        return sorted(found)

    def _walk_identifiers(
        self, node: Node, names: set[str], found: set[str],
    ) -> None:
        if node.type == "identifier" and node.text:
            ident = node.text.decode()
            if ident in names:
                found.add(ident)
        for child in node.children:
            self._walk_identifiers(child, names, found)

    # ------------------------------------------------------------------
    # Large chunk splitting
    # ------------------------------------------------------------------

    def _split_large_chunk(
        self,
        parent_chunk: CodeChunk,
        node: Node,
        lang: Language,
        import_texts: list[str],
    ) -> list[CodeChunk]:
        """Split a chunk that exceeds max_chunk_tokens at method boundaries."""
        method_types = _METHOD_TYPES.get(lang, set())
        file_path = parent_chunk.file_path

        # Find method nodes inside the definition
        methods: list[Node] = []
        self._collect_child_nodes(node, method_types, methods)

        if not methods:
            # No inner structure — split by lines
            return self._split_by_lines(parent_chunk, import_texts)

        chunks: list[CodeChunk] = []
        for method_node in methods:
            content = method_node.text.decode() if method_node.text else ""
            if not content.strip():
                continue
            start_line = method_node.start_point[0] + 1
            end_line = method_node.end_point[0] + 1
            name = f"{parent_chunk.symbol_name}.{self._get_node_name(method_node)}"
            docstring = self._get_docstring(method_node, lang)
            references = self._extract_references(method_node, import_texts) if import_texts else []

            chunks.append(CodeChunk(
                chunk_id=_chunk_id(file_path, start_line, end_line),
                file_path=file_path,
                language=parent_chunk.language,
                symbol_name=name,
                symbol_kind=SymbolKind.METHOD,
                content=content,
                start_line=start_line,
                end_line=end_line,
                docstring=docstring,
                imports=import_texts if self._config.include_imports else [],
                references=references,
                parent_symbol=parent_chunk.symbol_name,
            ))

        # If method splitting produced nothing, fall back to line splitting
        return chunks if chunks else self._split_by_lines(parent_chunk, import_texts)

    def _split_by_lines(
        self, chunk: CodeChunk, import_texts: list[str],
    ) -> list[CodeChunk]:
        """Split a chunk into line-based sub-chunks respecting token limits."""
        lines = chunk.content.splitlines(keepends=True)
        max_tokens = self._config.max_chunk_tokens
        sub_chunks: list[CodeChunk] = []
        buf: list[str] = []
        buf_start = chunk.start_line

        for i, line in enumerate(lines):
            buf.append(line)
            if _estimate_tokens("".join(buf)) >= max_tokens:
                content = "".join(buf)
                end_line = buf_start + len(buf) - 1
                sub_chunks.append(CodeChunk(
                    chunk_id=_chunk_id(chunk.file_path, buf_start, end_line),
                    file_path=chunk.file_path,
                    language=chunk.language,
                    symbol_name=chunk.symbol_name,
                    symbol_kind=chunk.symbol_kind,
                    content=content,
                    start_line=buf_start,
                    end_line=end_line,
                    docstring=chunk.docstring if not sub_chunks else "",
                    imports=import_texts if self._config.include_imports else [],
                    parent_symbol=chunk.parent_symbol,
                ))
                buf = []
                buf_start = chunk.start_line + i + 1

        # Flush remainder
        if buf:
            content = "".join(buf)
            end_line = buf_start + len(buf) - 1
            if _estimate_tokens(content) >= self._config.min_chunk_tokens:
                sub_chunks.append(CodeChunk(
                    chunk_id=_chunk_id(chunk.file_path, buf_start, end_line),
                    file_path=chunk.file_path,
                    language=chunk.language,
                    symbol_name=chunk.symbol_name,
                    symbol_kind=chunk.symbol_kind,
                    content=content,
                    start_line=buf_start,
                    end_line=end_line,
                    imports=import_texts if self._config.include_imports else [],
                    parent_symbol=chunk.parent_symbol,
                ))

        return sub_chunks if sub_chunks else [chunk]

    @staticmethod
    def _collect_child_nodes(
        node: Node, types: set[str], result: list[Node],
    ) -> None:
        """Recursively collect child nodes matching the given types."""
        for child in node.children:
            if child.type in types:
                result.append(child)
            else:
                ASTChunker._collect_child_nodes(child, types, result)

    # ------------------------------------------------------------------
    # Orphan code collection
    # ------------------------------------------------------------------

    def _collect_orphans(
        self,
        file: IngestedFile,
        root: Node,
        covered_ranges: list[tuple[int, int]],
        import_texts: list[str],
    ) -> list[CodeChunk]:
        """Collect code that doesn't belong to any definition or import."""
        source = file.content.encode()
        file_path = str(file.path)
        covered = sorted(covered_ranges)

        orphan_ranges: list[tuple[int, int]] = []
        pos = 0
        for start, end in covered:
            if pos < start:
                orphan_ranges.append((pos, start))
            pos = max(pos, end)
        if pos < len(source):
            orphan_ranges.append((pos, len(source)))

        chunks: list[CodeChunk] = []
        for start_byte, end_byte in orphan_ranges:
            text = source[start_byte:end_byte].decode(errors="replace").strip()
            if not text or _estimate_tokens(text) < self._config.min_chunk_tokens:
                continue

            # Compute line numbers from byte offsets
            start_line = source[:start_byte].count(b"\n") + 1
            end_line = source[:end_byte].count(b"\n") + 1

            chunks.append(CodeChunk(
                chunk_id=_chunk_id(file_path, start_line, end_line),
                file_path=file_path,
                language=file.language,
                symbol_name=file_path,
                symbol_kind=SymbolKind.MODULE,
                content=text,
                start_line=start_line,
                end_line=end_line,
                imports=import_texts if self._config.include_imports else [],
            ))

        return chunks
