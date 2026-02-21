"""Intent classification using keyword heuristics."""

from __future__ import annotations

import logging
import re

from gitrag.core.types import QueryIntent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern definitions per intent
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[QueryIntent, list[str]]] = [
    (QueryIntent.EXPLAIN, [
        r"\bexplain\b",
        r"\bwhat does\b",
        r"\bhow does\b",
        r"\bwhat is\b",
        r"\bdescribe\b",
        r"\bwalk me through\b",
    ]),
    (QueryIntent.BUG_FINDING, [
        r"\bbug\b",
        r"\berror\b",
        r"\bissue\b",
        r"\bwrong\b",
        r"\bfix\b",
        r"\bbroken\b",
        r"\bcrash\b",
        r"\bfail\b",
    ]),
    (QueryIntent.ARCHITECTURE, [
        r"\barchitecture\b",
        r"\bdesign\b",
        r"\bstructure\b",
        r"\boverview\b",
        r"\bhow is\b.*\borganized\b",
        r"\bpattern\b",
    ]),
    (QueryIntent.REFACTOR, [
        r"\brefactor\b",
        r"\bimprove\b",
        r"\bclean up\b",
        r"\bsimplify\b",
        r"\boptimize\b",
        r"\bbetter way\b",
    ]),
    (QueryIntent.DEPENDENCY, [
        r"\bdepend\w*\b",
        r"\bimport\b",
        r"\brequire\b",
        r"\buses\b",
        r"\bcalls\b",
        r"\breferenced by\b",
        r"\bused by\b",
    ]),
    (QueryIntent.SEARCH, [
        r"\bfind\b",
        r"\bwhere\b",
        r"\blocate\b",
        r"\bsearch\b",
        r"\bshow me\b",
        r"\blist\b",
    ]),
    (QueryIntent.HOW_TO, [
        r"\bhow to\b",
        r"\bhow can\b",
        r"\bhow do I\b",
        r"\bexample\b",
        r"\busage\b",
    ]),
]

_COMPILED_PATTERNS: list[tuple[QueryIntent, list[re.Pattern[str]]] ] = [
    (intent, [re.compile(p, re.IGNORECASE) for p in patterns])
    for intent, patterns in _INTENT_PATTERNS
]

_RETRIEVAL_DEPTHS: dict[QueryIntent, int] = {
    QueryIntent.ARCHITECTURE: 20,
    QueryIntent.BUG_FINDING: 15,
    QueryIntent.DEPENDENCY: 15,
    QueryIntent.EXPLAIN: 10,
    QueryIntent.REFACTOR: 10,
    QueryIntent.HOW_TO: 8,
    QueryIntent.SEARCH: 10,
    QueryIntent.UNKNOWN: 10,
}


class IntentClassifier:
    """Lightweight intent classifier based on keyword/pattern matching."""

    def classify(self, query: str) -> QueryIntent:
        """Classify the user query into a :class:`QueryIntent`.

        Patterns are evaluated in priority order; the first match wins.
        """
        for intent, compiled in _COMPILED_PATTERNS:
            for pattern in compiled:
                if pattern.search(query):
                    logger.debug("Classified query as %s (matched %s)", intent, pattern.pattern)
                    return intent
        logger.debug("No intent pattern matched — defaulting to UNKNOWN")
        return QueryIntent.UNKNOWN

    @staticmethod
    def get_retrieval_depth(intent: QueryIntent) -> int:
        """Return the recommended number of retrieval results for *intent*."""
        return _RETRIEVAL_DEPTHS.get(intent, 10)
