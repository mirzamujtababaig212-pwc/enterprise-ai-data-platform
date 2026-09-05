"""Deterministic mock provider for local and integration testing."""

from __future__ import annotations

from typing import Any

from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.llm_gateway.providers.base_provider import BaseProvider


class MockProvider(BaseProvider):
    """A zero-cost deterministic LLM provider.

    This provider never calls an external API.

    It is intended for:
    - local development
    - Docker smoke tests
    - integration tests
    - authentication testing
    - routing testing
    - metrics testing
    - observability testing
    """

    name = "mock"

    CHAT_MODELS = [
        "mock-gpt",
    ]

    EMBEDDING_MODELS = [
        "mock-embedding",
    ]

    STREAM_MODELS = [
        "mock-gpt",
    ]

    def supported_chat_models(self) -> list[str]:
        """Return supported chat models."""
        return list(self.CHAT_MODELS)

    def supported_embedding_models(self) -> list[str]:
        """Return supported embedding models."""
        return list(self.EMBEDDING_MODELS)

    def supported_stream_models(self) -> list[str]:
        """Return supported streaming models."""
        return list(self.STREAM_MODELS)

    async def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return a deterministic chat response or an opt-in tool call."""

        prompt = str(request.get("prompt", ""))

        tool_call = self._build_mock_tool_call(request)

        if tool_call is not None:
            return {
                "provider": self.name,
                "model": str(request.get("model", "mock-gpt")),
                "reply": "",
                "usage": {
                    "input_tokens": self._estimate_tokens(prompt),
                    "output_tokens": 0,
                },
                "tool_calls": [tool_call],
            }

        grounded_reply = self._build_mock_grounded_response(request)

        if grounded_reply is not None:
            reply = grounded_reply
        else:
            reply = f"Mock response: {prompt}"

        return {
            "provider": self.name,
            "model": str(request.get("model", "mock-gpt")),
            "reply": reply,
            "usage": {
                "input_tokens": self._estimate_tokens(prompt),
                "output_tokens": self._estimate_tokens(reply),
            },
            "tool_calls": [],
        }

    @staticmethod
    def _build_mock_tool_call(
        request: dict[str, Any],
    ) -> AgentToolCall | None:
        """Build an opt-in deterministic RAG tool call for agent tests."""

        metadata = request.get("metadata")

        if not isinstance(metadata, dict):
            return None

        if metadata.get("mock_tool_call") != "rag.search":
            return None

        tools = request.get("tools")

        if not isinstance(tools, list):
            return None

        if not any(isinstance(tool, dict) and tool.get("name") == "rag.search" for tool in tools):
            return None

        messages = request.get("messages", [])

        if not isinstance(messages, list):
            return None

        if any(isinstance(message, dict) and message.get("role") == "tool" for message in messages):
            return None

        return AgentToolCall(
            call_id="mock-rag-search-1",
            name="rag.search",
            arguments={
                "query": str(request.get("prompt", "")),
                "top_k": 3,
            },
        )

    @staticmethod
    def _build_mock_grounded_response(
        request: dict[str, Any],
    ) -> str | None:
        """Build a deterministic response from a previous RAG tool result."""

        messages = request.get("messages")

        if not isinstance(messages, list):
            return None

        for message in reversed(messages):
            if not isinstance(message, dict):
                continue

            if message.get("role") != "tool":
                continue

            content = message.get("content")

            if not isinstance(content, str):
                return None

            try:
                import json

                payload = json.loads(content)
            except (TypeError, ValueError):
                return None

            if not isinstance(payload, dict):
                return None

            if payload.get("tool_name") != "rag.search":
                return None

            if payload.get("success") is not True:
                return None

            output = payload.get("output")

            if not isinstance(output, dict):
                return None

            results = output.get("results")

            if not isinstance(results, list) or not results:
                return None

            first_result = results[0]

            if not isinstance(first_result, dict):
                return None

            retrieved_content = first_result.get("content")

            if not isinstance(retrieved_content, str):
                return None

            return (
                "Mock grounded response based on retrieved enterprise "
                f"context: {retrieved_content}"
            )

        return None

    async def embeddings(self, request: dict[str, Any]) -> list[float]:
        """Return a deterministic embedding vector."""

        text = str(request.get("text", ""))

        # Deterministic vector based on the input text.
        checksum = sum(ord(character) for character in text)

        return [
            round((checksum % 100) / 100.0, 4),
            round((len(text) % 100) / 100.0, 4),
            0.1234,
            0.5678,
        ]

    async def stream(self, request: dict[str, Any]):
        """Return deterministic streaming chunks."""

        prompt = str(request.get("prompt", ""))

        response = f"Mock response: {prompt}"

        for word in response.split():
            yield f"{word} "

    async def health_check(self) -> dict[str, Any]:
        """Return provider health."""

        return {
            "status": "healthy",
            "provider": self.name,
        }

    async def list_models(self) -> list[str]:
        """Return supported models."""

        return [
            *self.CHAT_MODELS,
            *self.EMBEDDING_MODELS,
        ]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Simple deterministic token estimate."""

        if not text:
            return 0

        return max(1, len(text.split()))
