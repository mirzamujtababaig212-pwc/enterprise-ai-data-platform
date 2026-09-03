from __future__ import annotations

from collections.abc import Sequence

from ai_platform.llm_gateway.routing.router import Router, router as default_router


class GatewayEmbeddingService:
    """
    RAG embedding service backed by the Enterprise LLM Gateway.

    RAG depends only on this adapter and the EmbeddingService contract.
    Provider selection and fallback remain the responsibility of the
    Enterprise LLM Gateway.
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

    async def embed(
        self,
        text: str,
    ) -> Sequence[float]:
        if not text.strip():
            raise ValueError("Text must not be empty.")

        return await self.gateway_router.route_embeddings(
            {
                "provider": self.provider,
                "model": self.model,
                "text": text,
            }
        )
