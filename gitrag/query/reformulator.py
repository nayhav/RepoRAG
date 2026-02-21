"""Query reformulation for follow-up queries in multi-turn conversations."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_CAMEL_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_FILE_PATH_RE = re.compile(r"[\w./\\-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|c|cpp|h|hpp|md|yaml|yml|toml|json)\b")
_QUOTED_RE = re.compile(r"""['"`]([^'"`]+)['"`]""")

_FOLLOWUP_PRONOUNS = re.compile(
    r"^\s*(?:it|that|this|they|those|these|the same|its|their)\b",
    re.IGNORECASE,
)

_MIN_SELF_CONTAINED_WORDS = 5


class QueryReformulator:
    """Heuristic-based query reformulator for multi-turn conversations."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reformulate(self, query: str, conversation_history: list[dict]) -> str:
        """Return a self-contained version of *query*.

        If the query already looks self-contained it is returned unchanged.
        Otherwise context from the most recent conversation turns is prepended
        to produce a standalone query.

        Parameters
        ----------
        query:
            The user's latest query.
        conversation_history:
            List of dicts with keys ``"role"`` (``"user"`` / ``"assistant"``)
            and ``"content"``.
        """
        if not self.is_followup(query):
            return query

        if not conversation_history:
            return query

        context = self._extract_recent_context(conversation_history)
        if not context:
            return query

        reformulated = f"Regarding {context}: {query}"
        logger.debug("Reformulated query: %s", reformulated)
        return reformulated

    def extract_entities(self, query: str) -> list[str]:
        """Extract potential code entities from *query*.

        Recognises:
        - CamelCase class names
        - snake_case function/variable names
        - File paths (with common extensions)
        - Quoted strings
        """
        entities: list[str] = []

        entities.extend(_CAMEL_CASE_RE.findall(query))
        entities.extend(_SNAKE_CASE_RE.findall(query))
        entities.extend(_FILE_PATH_RE.findall(query))
        entities.extend(_QUOTED_RE.findall(query))

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique.append(entity)
        return unique

    def is_followup(self, query: str) -> bool:
        """Heuristic check whether *query* is a follow-up question.

        A query is considered a follow-up when:
        - It starts with a pronoun (it, that, this, …)
        - It is very short (fewer than 5 words)
        - It lacks any recognisable code entity
        """
        if _FOLLOWUP_PRONOUNS.search(query):
            return True

        has_entities = bool(self.extract_entities(query))

        if len(query.split()) < _MIN_SELF_CONTAINED_WORDS and not has_entities:
            return True

        if not has_entities:
            return True

        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_recent_context(
        self,
        history: list[dict],
        max_turns: int = 3,
    ) -> str:
        """Pull key topics from the most recent conversation turns."""
        recent = history[-max_turns:]
        topics: list[str] = []
        for turn in reversed(recent):
            content = turn.get("content", "")
            entities = self.extract_entities(content)
            if entities:
                topics.extend(entities)
            elif turn.get("role") == "user" and content:
                # Fall back to the raw user message (truncated)
                topics.append(content[:120])

            if len(topics) >= 5:
                break

        if not topics:
            # Last resort: use the latest user message
            for turn in reversed(recent):
                if turn.get("role") == "user":
                    return turn.get("content", "")[:120]
            return ""

        return ", ".join(dict.fromkeys(topics))  # deduplicate, preserve order
