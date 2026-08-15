"""Deterministic mock provider for local and integration testing."""

from __future__ import annotations

from typing import Any
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
        """Return a deterministic chat response."""

        prompt = str(request.get("prompt", ""))

        return {
            "provider": self.name,
            "model": str(request.get("model", "mock-gpt")),
            "reply": f"Mock response: {prompt}",
            "usage": {
                "input_tokens": self._estimate_tokens(prompt),
                "output_tokens": self._estimate_tokens(f"Mock response: {prompt}"),
            },
        }

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
