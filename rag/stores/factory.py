from __future__ import annotations

from qdrant_client import AsyncQdrantClient

from common.config.settings import Settings
from rag.contracts import VectorStore
from rag.stores.in_memory import InMemoryVectorStore
from rag.stores.faiss import FAISSVectorStore
from rag.stores.postgres import PostgreSQLVectorStore
from rag.stores.qdrant import QdrantVectorStore


class VectorStoreFactory:
    """
    Constructs the configured RAG vector-store backend.

    In-memory storage remains the default so existing local/test behavior
    is unchanged unless VECTOR_STORE_BACKEND is explicitly set to faiss,
    qdrant, or postgres.
    """

    @staticmethod
    def create() -> VectorStore:
        backend = Settings.vector_store.BACKEND

        if backend == "in_memory":
            return InMemoryVectorStore()

        if backend == "faiss":
            return FAISSVectorStore()

        if backend == "qdrant":
            client = AsyncQdrantClient(
                url=Settings.qdrant.URL,
                api_key=Settings.qdrant.API_KEY,
                timeout=Settings.qdrant.TIMEOUT,
            )
            return QdrantVectorStore(
                client=client,
                collection_name=Settings.qdrant.COLLECTION,
            )

        if backend == "postgres":
            return PostgreSQLVectorStore()

        raise ValueError(
            "Unsupported vector-store backend: "
            f"{backend!r}. Expected 'in_memory', 'faiss', 'qdrant', or 'postgres'."
        )
