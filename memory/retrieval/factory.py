from __future__ import annotations

from memory.retrieval.contracts import MemoryRetriever
from memory.retrieval.hybrid import HybridMemoryRetriever
from memory.retrieval.lexical import LexicalMemoryRetriever
from memory.retrieval.postgres_lexical import PostgreSQLLexicalMemoryRetriever
from memory.retrieval.postgres_semantic import PostgreSQLSemanticMemoryRetriever
from rag.contracts import EmbeddingService


class MemoryRetrieverFactory:
    @staticmethod
    def create(
        *,
        backend: str,
        memory_store,
        embedding_service: EmbeddingService | None = None,
    ) -> MemoryRetriever:
        normalized_backend = backend.strip().lower()

        if normalized_backend == "in_memory":
            return LexicalMemoryRetriever(memory_store)

        if normalized_backend == "postgres":
            if embedding_service is None:
                raise ValueError(
                    "An embedding service is required for PostgreSQL " "hybrid memory retrieval."
                )

            semantic_retriever = PostgreSQLSemanticMemoryRetriever(
                embedding_service=embedding_service,
            )
            lexical_retriever = PostgreSQLLexicalMemoryRetriever()

            return HybridMemoryRetriever(
                semantic_retriever=semantic_retriever,
                lexical_retriever=lexical_retriever,
            )

        raise ValueError(
            "Unsupported memory-store backend: "
            f"{normalized_backend!r}. Expected 'in_memory' or 'postgres'."
        )
