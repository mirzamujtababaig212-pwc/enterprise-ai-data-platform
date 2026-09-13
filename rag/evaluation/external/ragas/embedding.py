from __future__ import annotations

from rag.contracts import EmbeddingService
from ragas.embeddings.base import BaseRagasEmbedding


class GatewayRagasEmbedding(BaseRagasEmbedding):
    """Adapt the platform EmbeddingService to the RAGAS embedding contract."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self._embedding_service = embedding_service

    async def aembed_text(self, text: str, **kwargs: object) -> list[float]:
        return list(await self._embedding_service.embed(text))

    async def aembed_texts(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> list[list[float]]:
        return [await self.aembed_text(text) for text in texts]

    def embed_text(self, text: str, **kwargs: object) -> list[float]:
        raise TypeError("GatewayRagasEmbedding is asynchronous; use aembed_text().")

    def embed_texts(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> list[list[float]]:
        raise TypeError("GatewayRagasEmbedding is asynchronous; use aembed_texts().")
