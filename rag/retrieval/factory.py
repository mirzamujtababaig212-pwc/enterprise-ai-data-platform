from __future__ import annotations

from rag.contracts import Retriever
from rag.evaluation.lineage import (
    HybridRetrievalConfiguration,
    RerankerConfiguration,
    RetrievalEvaluationArtifact,
)
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.lexical import PostgreSQLLexicalRetriever
from rag.retrieval.reranker import CrossEncoderReranker, RerankingRetriever
from rag.retrieval.retriever import SemanticRetriever


class RAGRetrieverFactory:
    @staticmethod
    def build_retrieval_artifact(
        retriever: Retriever,
    ) -> RetrievalEvaluationArtifact:
        """Build immutable provenance from the instantiated retriever graph."""

        if isinstance(retriever, RerankingRetriever):
            underlying_artifact = RAGRetrieverFactory.build_retrieval_artifact(
                retriever.retriever,
            )

            reranker = retriever.reranker

            if isinstance(reranker, CrossEncoderReranker):
                reranker_configuration = RerankerConfiguration(
                    type=type(reranker).__name__,
                    model_id=reranker.model_id,
                    onnx_filename=reranker.onnx_filename,
                    max_length=reranker.max_length,
                    candidate_k=retriever.candidate_k,
                )
            else:
                reranker_configuration = RerankerConfiguration(
                    type=type(reranker).__name__,
                    candidate_k=retriever.candidate_k,
                )

            return RetrievalEvaluationArtifact(
                retriever_type=type(retriever).__name__,
                vector_store_type=underlying_artifact.vector_store_type,
                hybrid_configuration=underlying_artifact.hybrid_configuration,
                reranker_configuration=reranker_configuration,
            )

        if isinstance(retriever, HybridRetriever):
            semantic_retriever = retriever.semantic_retriever

            if not isinstance(semantic_retriever, SemanticRetriever):
                raise ValueError(
                    "HybridRetriever provenance requires a SemanticRetriever "
                    "as its semantic component."
                )

            return RetrievalEvaluationArtifact(
                retriever_type=type(retriever).__name__,
                vector_store_type=type(semantic_retriever.vector_store).__name__,
                hybrid_configuration=HybridRetrievalConfiguration(
                    candidate_k=retriever.candidate_k,
                    rrf_k=retriever.rrf_k,
                    semantic_weight=retriever.semantic_weight,
                    lexical_weight=retriever.lexical_weight,
                ),
            )

        if isinstance(retriever, SemanticRetriever):
            return RetrievalEvaluationArtifact(
                retriever_type=type(retriever).__name__,
                vector_store_type=type(retriever.vector_store).__name__,
            )

        raise ValueError(
            "Unsupported retriever type for runtime provenance extraction: "
            f"{type(retriever).__name__!r}."
        )

    @staticmethod
    def create(
        *,
        backend: str,
        semantic_retriever: SemanticRetriever,
        lexical_retriever: PostgreSQLLexicalRetriever | None = None,
        reranker: str = "none",
        reranker_model_id: str = CrossEncoderReranker.DEFAULT_MODEL_ID,
        reranker_onnx_filename: str = CrossEncoderReranker.DEFAULT_ONNX_FILENAME,
        reranker_max_length: int = 8192,
        reranker_candidate_k: int = 20,
    ) -> Retriever:
        normalized_backend = backend.strip().lower()
        normalized_reranker = reranker.strip().lower()

        if normalized_reranker not in {"none", "cross_encoder"}:
            raise ValueError(
                "Unsupported RAG reranker: "
                f"{normalized_reranker!r}. Expected 'none' or 'cross_encoder'."
            )

        if reranker_max_length <= 0:
            raise ValueError("reranker_max_length must be greater than zero.")

        if reranker_candidate_k <= 0:
            raise ValueError("reranker_candidate_k must be greater than zero.")

        if normalized_backend == "in_memory":
            if normalized_reranker == "cross_encoder":
                raise ValueError("Cross-encoder RAG reranking requires a hybrid RAG backend.")

            return semantic_retriever

        if normalized_backend in {"postgres", "qdrant"}:
            if lexical_retriever is None:
                raise ValueError(
                    "A PostgreSQL lexical retriever is required for hybrid RAG backends."
                )

            hybrid_candidate_k = (
                reranker_candidate_k if normalized_reranker == "cross_encoder" else 5
            )

            retriever: Retriever = HybridRetriever(
                semantic_retriever=semantic_retriever,
                lexical_retriever=lexical_retriever,
                candidate_k=hybrid_candidate_k,
                rrf_k=60,
                semantic_weight=1.0,
                lexical_weight=0.5,
            )

            if normalized_reranker == "cross_encoder":
                reranker_instance = CrossEncoderReranker(
                    model_id=reranker_model_id,
                    onnx_filename=reranker_onnx_filename,
                    max_length=reranker_max_length,
                )

                return RerankingRetriever(
                    retriever,
                    reranker_instance,
                    candidate_k=reranker_candidate_k,
                )

            return retriever

        raise ValueError(
            "Unsupported RAG retrieval backend: "
            f"{normalized_backend!r}. Expected 'in_memory', 'postgres', or 'qdrant'."
        )
