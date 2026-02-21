"""Fallback chunker for documentation and config files."""

from __future__ import annotations

import hashlib
import logging
import re

from gitrag.config import ChunkingConfig
from gitrag.core.types import CodeChunk, IngestedFile, Language, SymbolKind

from .base import BaseChunker

log = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _chunk_id(file_path: str, start_line: int, end_line: int) -> str:
    raw = f"{file_path}:{start_line}:{end_line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class TextChunker(BaseChunker):
    """Split non-code files (markdown, docs, config) into chunks."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._config = config or ChunkingConfig()

    def chunk(self, file: IngestedFile) -> list[CodeChunk]:
        if file.language in (Language.MARKDOWN, Language.RST):
            return self._chunk_markdown(file)
        return self._chunk_plain(file)

    # ------------------------------------------------------------------
    # Markdown / RST: split on headers
    # ------------------------------------------------------------------

    def _chunk_markdown(self, file: IngestedFile) -> list[CodeChunk]:
        lines = file.content.splitlines(keepends=True)
        sections: list[tuple[str, int, int]] = []
        current_start = 0
        header_pattern = re.compile(r"^#{1,6}\s")

        for i, line in enumerate(lines):
            if header_pattern.match(line) and i > current_start:
                sections.append(self._join_lines(lines, current_start, i))
                current_start = i

        # Final section
        if current_start < len(lines):
            sections.append(self._join_lines(lines, current_start, len(lines)))

        return self._sections_to_chunks(file, sections, SymbolKind.DOCUMENTATION)

    # ------------------------------------------------------------------
    # Plain text / config: split by blank-line-separated paragraphs
    # ------------------------------------------------------------------

    def _chunk_plain(self, file: IngestedFile) -> list[CodeChunk]:
        lines = file.content.splitlines(keepends=True)
        sections: list[tuple[str, int, int]] = []
        current_start = 0

        for i, line in enumerate(lines):
            if line.strip() == "" and i > current_start:
                text = "".join(lines[current_start:i])
                if text.strip():
                    sections.append((text, current_start, i))
                current_start = i + 1

        if current_start < len(lines):
            text = "".join(lines[current_start:])
            if text.strip():
                sections.append((text, current_start, len(lines)))

        kind = SymbolKind.DOCUMENTATION if file.language in (
            Language.MARKDOWN, Language.RST, Language.TEXT,
        ) else SymbolKind.MODULE

        return self._sections_to_chunks(file, sections, kind)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _join_lines(
        lines: list[str], start: int, end: int,
    ) -> tuple[str, int, int]:
        return "".join(lines[start:end]), start, end

    def _sections_to_chunks(
        self,
        file: IngestedFile,
        sections: list[tuple[str, int, int]],
        kind: SymbolKind,
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        max_tokens = self._config.max_chunk_tokens
        min_tokens = self._config.min_chunk_tokens
        file_path = str(file.path)

        merged_text = ""
        merged_start = 0
        merged_end = 0

        for text, start, end in sections:
            candidate = merged_text + text
            if _estimate_tokens(candidate) > max_tokens and merged_text:
                # Flush accumulated text
                chunks.append(self._make_chunk(
                    file_path, file.language, kind,
                    merged_text, merged_start, merged_end,
                ))
                merged_text = text
                merged_start = start
                merged_end = end
            else:
                if not merged_text:
                    merged_start = start
                merged_text = candidate
                merged_end = end

        # Flush remainder
        if merged_text.strip() and _estimate_tokens(merged_text) >= min_tokens:
            chunks.append(self._make_chunk(
                file_path, file.language, kind,
                merged_text, merged_start, merged_end,
            ))

        if not chunks and file.content.strip():
            # File too small to split — emit a single chunk
            total_lines = file.content.count("\n") + 1
            chunks.append(self._make_chunk(
                file_path, file.language, kind,
                file.content, 0, total_lines,
            ))

        log.debug("TextChunker produced %d chunks for %s", len(chunks), file_path)
        return chunks

    @staticmethod
    def _make_chunk(
        file_path: str,
        language: Language,
        kind: SymbolKind,
        text: str,
        start: int,
        end: int,
    ) -> CodeChunk:
        start_line = start + 1  # 1-indexed
        end_line = end  # already past-the-end index, so this is inclusive
        return CodeChunk(
            chunk_id=_chunk_id(file_path, start_line, end_line),
            file_path=file_path,
            language=language,
            symbol_name=file_path,
            symbol_kind=kind,
            content=text,
            start_line=start_line,
            end_line=end_line,
        )
