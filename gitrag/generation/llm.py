"""LLM client interface using the OpenAI Python client library."""

from __future__ import annotations

import logging
import re

from openai import OpenAI

from gitrag.config import GenerationConfig
from gitrag.core.types import Citation, GeneratedAnswer, RetrievalResult

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper around the OpenAI chat-completions API.

    Works with both the OpenAI API and Ollama's OpenAI-compatible endpoint.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

        if config.provider == "ollama":
            base_url = config.base_url.rstrip("/") + "/v1"
            api_key = "ollama"
        else:
            base_url = config.base_url
            api_key = config.api_key

        self._client = OpenAI(base_url=base_url, api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        context: str,
        query: str,
        conversation_history: str = "",
        chunks: list[RetrievalResult] | None = None,
    ) -> GeneratedAnswer:
        """Send a chat-completion request and return a ``GeneratedAnswer``."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if conversation_history:
            messages.append({
                "role": "user",
                "content": f"Previous conversation:\n{conversation_history}",
            })
        messages.append({
            "role": "user",
            "content": f"Code Context:\n{context}\n\nQuestion: {query}",
        })

        try:
            response = self._client.chat.completions.create(
                model=self.config.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        except Exception:
            logger.warning("LLM request failed", exc_info=True)
            return GeneratedAnswer(
                content="Error: unable to reach the LLM. Please check the connection.",
                model=self.config.model,
            )

        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage

        citations = self._parse_citations(content, chunks or [])

        return GeneratedAnswer(
            content=content,
            citations=citations,
            model=self.config.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def health_check(self) -> bool:
        """Return *True* if the LLM endpoint is reachable."""
        try:
            self._client.models.list()
            return True
        except Exception:
            logger.warning("LLM health-check failed", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_citations(
        response_text: str,
        chunks: list[RetrievalResult],
    ) -> list[Citation]:
        """Extract ``[file:line]`` / ``[file:start-end]`` references from the response."""
        pattern = re.compile(r"\[([^:\[\]]+):(\d+)(?:-(\d+))?\]")

        chunk_map: dict[str, RetrievalResult] = {}
        for result in chunks:
            chunk_map[result.chunk.file_path] = result

        citations: list[Citation] = []
        seen: set[tuple[str, int, int]] = set()

        for match in pattern.finditer(response_text):
            file_path = match.group(1)
            start_line = int(match.group(2))
            end_line = int(match.group(3)) if match.group(3) else start_line

            key = (file_path, start_line, end_line)
            if key in seen:
                continue
            seen.add(key)

            symbol_name = ""
            chunk_id = ""
            snippet = ""
            matched = chunk_map.get(file_path)
            if matched:
                symbol_name = matched.chunk.symbol_name
                chunk_id = matched.chunk.chunk_id
                snippet = matched.chunk.content[:120]

            citations.append(Citation(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                symbol_name=symbol_name,
                chunk_id=chunk_id,
                snippet=snippet,
            ))

        return citations
