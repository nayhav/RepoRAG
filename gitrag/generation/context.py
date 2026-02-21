"""Context compression to fit within LLM context windows."""

from __future__ import annotations

from gitrag.core.types import RetrievalResult


class ContextCompressor:
    """Compress and deduplicate retrieval results to fit a token budget."""

    def __init__(self, max_context_tokens: int) -> None:
        self.max_context_tokens = max_context_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Keep the highest-scored chunks that fit within the token budget.

        Strategy:
        1. Sort by *final_score* descending.
        2. Accumulate chunks until the budget is reached.
        3. If a chunk pushes over the budget, truncate its content (keep first
           N lines) so it still partially fits.
        4. Return the chunks that fit.

        Token estimation: ``len(text) // 4``.
        """
        sorted_results = sorted(results, key=lambda r: r.final_score, reverse=True)

        kept: list[RetrievalResult] = []
        tokens_used = 0

        for result in sorted_results:
            chunk_tokens = len(result.chunk.content) // 4
            if tokens_used + chunk_tokens <= self.max_context_tokens:
                kept.append(result)
                tokens_used += chunk_tokens
            else:
                remaining = self.max_context_tokens - tokens_used
                if remaining > 0:
                    lines = result.chunk.content.splitlines(keepends=True)
                    truncated_lines: list[str] = []
                    truncated_tokens = 0
                    for line in lines:
                        line_tokens = len(line) // 4
                        if truncated_tokens + line_tokens > remaining:
                            break
                        truncated_lines.append(line)
                        truncated_tokens += line_tokens
                    if truncated_lines:
                        result.chunk.content = "".join(truncated_lines)
                        result.chunk.end_line = (
                            result.chunk.start_line + len(truncated_lines) - 1
                        )
                        kept.append(result)
                        tokens_used += truncated_tokens
                break

        return kept

    def deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Remove results with highly overlapping content (>80% line overlap).

        When two chunks in the same file overlap by more than 80% of lines,
        only the higher-scored one is kept.
        """
        sorted_results = sorted(results, key=lambda r: r.final_score, reverse=True)

        kept: list[RetrievalResult] = []
        for result in sorted_results:
            if not self._overlaps_any(result, kept):
                kept.append(result)
        return kept

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _overlaps_any(candidate: RetrievalResult, kept: list[RetrievalResult]) -> bool:
        """Return *True* if *candidate* overlaps >80% with any kept result."""
        c = candidate.chunk
        c_lines = set(range(c.start_line, c.end_line + 1))
        if not c_lines:
            return False

        for existing in kept:
            e = existing.chunk
            if e.file_path != c.file_path:
                continue
            e_lines = set(range(e.start_line, e.end_line + 1))
            overlap = len(c_lines & e_lines)
            if overlap / len(c_lines) > 0.8:
                return True
        return False
