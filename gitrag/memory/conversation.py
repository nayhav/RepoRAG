"""Conversation memory with three-tier strategy.

Short-term buffer keeps recent turns in full.  A rolling summary
compresses older turns at a configurable interval.  Context-window
optimization builds an LLM-ready string within a token budget.
"""

from __future__ import annotations

import re

from gitrag.config import MemoryConfig
from gitrag.core.types import Conversation, ConversationTurn


class ConversationMemory:
    """Manages conversation state for multi-turn interactions."""

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(
        self,
        conversation: Conversation,
        role: str,
        content: str,
        cited_chunks: list[str] | None = None,
    ) -> None:
        """Add a turn and trigger summarization when needed."""
        conversation.add_turn(role, content, cited_chunks)

        # Enforce max_turns: drop the oldest turn when over the limit.
        while len(conversation.turns) > self._config.max_turns:
            conversation.turns.pop(0)

        # Trigger rolling summarization every summary_interval turns.
        total = len(conversation.turns)
        if (
            self._config.summary_interval > 0
            and total >= self._config.summary_interval
            and total % self._config.summary_interval == 0
        ):
            # Summarize all turns except the most recent summary_interval.
            older = conversation.turns[: -self._config.summary_interval]
            if older:
                new_summary = self.summarize_turns(older)
                if conversation.summary:
                    conversation.summary = (
                        conversation.summary + " " + new_summary
                    )
                else:
                    conversation.summary = new_summary

    def get_context_for_llm(
        self,
        conversation: Conversation,
        max_tokens: int = 2000,
    ) -> str:
        """Build a context string fitting within *max_tokens* (estimated).

        Includes the rolling summary (if any) followed by recent turns,
        truncating the oldest turns first when the budget is exceeded.
        Token estimation: ``len(text) // 4``.
        """
        parts: list[str] = []
        budget = max_tokens

        # 1. Include summary if it exists.
        if conversation.summary:
            summary_text = f"Summary of earlier conversation:\n{conversation.summary}\n"
            summary_tokens = len(summary_text) // 4
            if summary_tokens < budget:
                parts.append(summary_text)
                budget -= summary_tokens

        # 2. Format recent turns and fit within remaining budget.
        formatted_turns: list[str] = []
        for turn in conversation.turns:
            label = "User" if turn.role == "user" else "Assistant"
            formatted_turns.append(f"{label}: {turn.content}")

        # Add turns newest-first, then reverse so order is chronological.
        selected: list[str] = []
        for ft in reversed(formatted_turns):
            ft_tokens = len(ft) // 4
            if ft_tokens > budget:
                break
            selected.append(ft)
            budget -= ft_tokens

        selected.reverse()
        if selected:
            parts.append("\n".join(selected))

        return "\n".join(parts)

    def summarize_turns(self, turns: list[ConversationTurn]) -> str:
        """Create a concise extractive summary of *turns*.

        Takes the first sentence of each assistant response and joins them.
        """
        sentences: list[str] = []
        for turn in turns:
            if turn.role != "assistant":
                continue
            first = _first_sentence(turn.content)
            if first:
                sentences.append(first)

        if not sentences:
            return ""
        return "Previous discussion covered: " + " ".join(sentences)

    def get_recent_turns(
        self,
        conversation: Conversation,
        n: int = 3,
    ) -> list[ConversationTurn]:
        """Return the last *n* turns."""
        return conversation.turns[-n:]

    def get_search_context(self, conversation: Conversation) -> str:
        """Extract key terms from the last 2-3 turns for query reformulation.

        Pulls out nouns and code-style identifiers (``snake_case``,
        ``CamelCase``, dotted paths) from recent user and assistant turns.
        """
        recent = conversation.turns[-3:]
        text = " ".join(t.content for t in recent)
        terms = _extract_identifiers(text)
        return " ".join(terms)

    def clear(self, conversation: Conversation) -> None:
        """Reset conversation state."""
        conversation.turns.clear()
        conversation.summary = ""


# ----------------------------------------------------------------------
# Private helpers
# ----------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"([^.!?\n]+[.!?])")

# Matches snake_case, CamelCase, UPPER_CASE, and dotted.paths identifiers
# that are at least 2 characters long.
_IDENT_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9]+|[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+|[a-z_][a-z0-9_]{2,}|[A-Z][A-Z0-9_]{2,})\b"
)

# Common English stop-words to filter out from extracted terms.
_STOP_WORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can",
    "had", "her", "was", "one", "our", "out", "has", "have", "from",
    "been", "some", "them", "than", "this", "that", "they", "with",
    "will", "each", "make", "like", "does", "into", "over", "such",
    "also", "more", "other", "about", "which", "their", "there",
    "would", "could", "should", "what", "when", "where", "how",
    "use", "used", "using",
})


def _first_sentence(text: str) -> str:
    """Return the first sentence from *text*, or the whole text if short."""
    text = text.strip()
    m = _SENTENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    # No sentence boundary found — return up to 120 chars.
    return text[:120].strip()


def _extract_identifiers(text: str) -> list[str]:
    """Pull code identifiers and meaningful nouns from *text*."""
    matches = _IDENT_RE.findall(text)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        low = m.lower()
        if low in _STOP_WORDS or low in seen:
            continue
        seen.add(low)
        result.append(m)
    return result
