from __future__ import annotations

from collections.abc import Sequence

from rag.contracts import EmbeddingService, VectorStore
from rag.models import RetrievalResult


class SemanticRetriever:
    """
    Query-to-vector retrieval service.

    The retriever knows about embedding and vector-store contracts,
    but knows nothing about a specific embedding provider or database.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Sequence[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        embedding = await self.embedding_service.embed(query)

        return await self.vector_store.search(
            embedding,
            top_k=top_k,
        )
