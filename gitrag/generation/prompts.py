"""Prompt templates for different query intents."""

from __future__ import annotations

from gitrag.core.types import QueryIntent, RetrievalResult

_SYSTEM_PROMPT = (
    "You are GitRAG, a code-aware AI assistant that answers questions about a codebase.\n"
    "You MUST base your answers strictly on the provided code context.\n"
    "You MUST cite source files using [file:line] format.\n"
    "If the context is insufficient to answer, say so explicitly.\n"
    "Do NOT hallucinate or invent code that is not in the context."
)

_INTENT_INSTRUCTIONS: dict[QueryIntent, str] = {
    QueryIntent.EXPLAIN: (
        "Explain the following code element in detail. "
        "Cite all relevant source locations."
    ),
    QueryIntent.BUG_FINDING: (
        "Analyze the code for potential bugs or issues. "
        "Be specific about what could go wrong and cite locations."
    ),
    QueryIntent.ARCHITECTURE: (
        "Describe the architecture and design patterns. "
        "Reference specific files and components."
    ),
    QueryIntent.REFACTOR: (
        "Suggest improvements to the code. "
        "Explain the rationale and cite current implementations."
    ),
    QueryIntent.DEPENDENCY: (
        "Trace the dependencies and relationships. "
        "Show the chain of imports/calls with file references."
    ),
    QueryIntent.SEARCH: (
        "Find and present the relevant code. Cite exact locations."
    ),
    QueryIntent.HOW_TO: (
        "Explain how to use or implement this. "
        "Provide examples from the codebase."
    ),
}


def build_system_prompt() -> str:
    """Return the system prompt."""
    return _SYSTEM_PROMPT


def build_context_prompt(results: list[RetrievalResult], max_tokens: int) -> str:
    """Format retrieved chunks as numbered context blocks.

    Chunks are ordered by *final_score* descending.  The lowest-scored chunks
    are dropped first when the token budget is exceeded.
    """
    sorted_results = sorted(results, key=lambda r: r.final_score, reverse=True)

    blocks: list[str] = []
    tokens_used = 0

    for i, result in enumerate(sorted_results, 1):
        chunk = result.chunk
        block = (
            f"[{i}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line} "
            f"({chunk.symbol_name})\n{chunk.content}"
        )
        block_tokens = len(block) // 4
        if tokens_used + block_tokens > max_tokens:
            break
        blocks.append(block)
        tokens_used += block_tokens

    return "\n\n".join(blocks)


def build_query_prompt(
    query: str,
    intent: QueryIntent,
    conversation_context: str = "",
) -> str:
    """Format the user query with intent-specific instructions."""
    instruction = _INTENT_INSTRUCTIONS.get(intent, "Cite source files.")

    parts: list[str] = []
    if conversation_context:
        parts.append(f"Conversation context:\n{conversation_context}")
    parts.append(instruction)
    parts.append(f"Question: {query}")
    return "\n\n".join(parts)
