from __future__ import annotations


from ai_platform.llm_gateway.routing.router import Router, router as default_router
from rag.models import EmbeddingIdentity, EmbeddingResult


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
    ) -> list[float]:
        result = await self.embed_with_metadata(text)
        return list(result.vector)

    async def embed_with_metadata(
        self,
        text: str,
    ) -> EmbeddingResult:
        if not text.strip():
            raise ValueError("Text must not be empty.")

        result = await self.gateway_router.route_embeddings_with_metadata(
            {
                "provider": self.provider,
                "model": self.model,
                "text": text,
            }
        )

        vector = tuple(float(value) for value in result.response)

        identity = EmbeddingIdentity(
            requested_provider=self.provider,
            requested_model=self.model,
            resolved_provider=result.provider_name,
            resolved_model=result.model_name or self.model,
            dimension=len(vector),
        )

        return EmbeddingResult(
            vector=vector,
            identity=identity,
        )
