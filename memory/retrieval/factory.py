from __future__ import annotations

from memory.retrieval.contracts import MemoryRetriever
from memory.retrieval.hybrid import HybridMemoryRetriever
from memory.retrieval.lexical import LexicalMemoryRetriever
from memory.retrieval.postgres_lexical import PostgreSQLLexicalMemoryRetriever
from memory.retrieval.postgres_semantic import PostgreSQLSemanticMemoryRetriever
from memory.retrieval.reranker import CrossEncoderMemoryReranker
from memory.retrieval.reranking import RerankingMemoryRetriever
from rag.contracts import EmbeddingService


class MemoryRetrieverFactory:
    @staticmethod
    def create(
        *,
        backend: str,
        memory_store,
        embedding_service: EmbeddingService | None = None,
        reranker: str = "none",
        reranker_model_id: str = CrossEncoderMemoryReranker.DEFAULT_MODEL_ID,
        reranker_onnx_filename: str = CrossEncoderMemoryReranker.DEFAULT_ONNX_FILENAME,
        reranker_max_length: int = 8192,
        reranker_candidate_k: int = 20,
    ) -> MemoryRetriever:
        normalized_backend = backend.strip().lower()
        normalized_reranker = reranker.strip().lower()

        if normalized_reranker not in {"none", "cross_encoder"}:
            raise ValueError(
                "Unsupported memory reranker: "
                f"{normalized_reranker!r}. Expected 'none' or 'cross_encoder'."
            )

        if reranker_max_length <= 0:
            raise ValueError("reranker_max_length must be greater than zero.")

        if reranker_candidate_k <= 0:
            raise ValueError("reranker_candidate_k must be greater than zero.")

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

            hybrid_candidate_k = (
                reranker_candidate_k if normalized_reranker == "cross_encoder" else 10
            )

            retriever: MemoryRetriever = HybridMemoryRetriever(
                semantic_retriever=semantic_retriever,
                lexical_retriever=lexical_retriever,
                candidate_k=hybrid_candidate_k,
            )

            if normalized_reranker == "cross_encoder":
                reranker_instance = CrossEncoderMemoryReranker(
                    model_id=reranker_model_id,
                    onnx_filename=reranker_onnx_filename,
                    max_length=reranker_max_length,
                )
                return RerankingMemoryRetriever(
                    retriever,
                    reranker_instance,
                    candidate_k=reranker_candidate_k,
                )

            return retriever

        raise ValueError(
            "Unsupported memory-store backend: "
            f"{normalized_backend!r}. Expected 'in_memory' or 'postgres'."
        )
