from __future__ import annotations

from typing import Any

from ai_platform.llm_gateway.routing.router import (
    Router,
    router as default_router,
)


class GatewayChatService:
    """
    RAG generation service backed by the Enterprise LLM Gateway.

    RAG does not communicate with model providers directly.
    Provider selection, validation, fallback, observability, and
    execution remain responsibilities of the LLM Gateway.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        gateway_router: Router | None = None,
    ) -> None:
        if not provider.strip():
            raise ValueError("provider must not be empty.")

        if not model.strip():
            raise ValueError("model must not be empty.")

        self.provider = provider
        self.model = model
        self.gateway_router = gateway_router or default_router

    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")

        request: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if user_id is not None:
            request["user_id"] = user_id

        return await self.gateway_router.route_chat(request)
