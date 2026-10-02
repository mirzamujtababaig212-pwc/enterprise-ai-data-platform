from __future__ import annotations

from dataclasses import dataclass

from rag.models import EmbeddingIdentity


@dataclass(frozen=True)
class HybridRetrievalConfiguration:
    """Immutable configuration provenance for weighted hybrid retrieval."""

    candidate_k: int
    rrf_k: int
    semantic_weight: float
    lexical_weight: float

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        if self.semantic_weight < 0.0:
            raise ValueError("semantic_weight must be greater than or equal to zero")

        if self.lexical_weight < 0.0:
            raise ValueError("lexical_weight must be greater than or equal to zero")

        if self.semantic_weight == 0.0 and self.lexical_weight == 0.0:
            raise ValueError("At least one retrieval weight must be greater than zero")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_k": self.candidate_k,
            "rrf_k": self.rrf_k,
            "semantic_weight": self.semantic_weight,
            "lexical_weight": self.lexical_weight,
        }


@dataclass(frozen=True)
class RerankerConfiguration:
    """Immutable configuration provenance for retrieval reranking."""

    type: str
    model_id: str | None = None
    onnx_filename: str | None = None
    max_length: int | None = None
    candidate_k: int | None = None

    def __post_init__(self) -> None:
        if not self.type.strip():
            raise ValueError("reranker type must not be empty")

        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("reranker max_length must be greater than zero")

        if self.candidate_k is not None and self.candidate_k <= 0:
            raise ValueError("reranker candidate_k must be greater than zero")

        if self.model_id is not None and not self.model_id.strip():
            raise ValueError("reranker model_id must not be empty")

        if self.onnx_filename is not None and not self.onnx_filename.strip():
            raise ValueError("reranker onnx_filename must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "model_id": self.model_id,
            "onnx_filename": self.onnx_filename,
            "max_length": self.max_length,
            "candidate_k": self.candidate_k,
        }


@dataclass(frozen=True)
class RetrievalEvaluationArtifact:
    """
    Immutable provenance for the retrieval implementation evaluated by a run.

    This describes the executable retrieval components without claiming an
    immutable production index or artifact version.
    """

    retriever_type: str
    vector_store_type: str
    hybrid_configuration: HybridRetrievalConfiguration | None = None
    reranker_configuration: RerankerConfiguration | None = None

    def __post_init__(self) -> None:
        if not self.retriever_type.strip():
            raise ValueError("retriever_type must not be empty")

        if not self.vector_store_type.strip():
            raise ValueError("vector_store_type must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "retriever_type": self.retriever_type,
            "vector_store_type": self.vector_store_type,
            "hybrid_configuration": (
                self.hybrid_configuration.as_dict()
                if self.hybrid_configuration is not None
                else None
            ),
            "reranker_configuration": (
                self.reranker_configuration.as_dict()
                if self.reranker_configuration is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RetrievalEvaluationArtifact":
        """Reconstruct persisted retrieval provenance."""

        if not isinstance(value, dict):
            raise ValueError("retrieval artifact must be a dictionary.")

        retriever_type = value.get("retriever_type")
        vector_store_type = value.get("vector_store_type")

        if not isinstance(retriever_type, str):
            raise ValueError("retrieval artifact retriever_type must be a string.")

        if not isinstance(vector_store_type, str):
            raise ValueError("retrieval artifact vector_store_type must be a string.")

        hybrid_data = value.get("hybrid_configuration")
        hybrid_configuration = None

        if hybrid_data is not None:
            if not isinstance(hybrid_data, dict):
                raise ValueError("hybrid_configuration must be a dictionary.")

            hybrid_configuration = HybridRetrievalConfiguration(
                candidate_k=hybrid_data["candidate_k"],
                rrf_k=hybrid_data["rrf_k"],
                semantic_weight=hybrid_data["semantic_weight"],
                lexical_weight=hybrid_data["lexical_weight"],
            )

        reranker_data = value.get("reranker_configuration")
        reranker_configuration = None

        if reranker_data is not None:
            if not isinstance(reranker_data, dict):
                raise ValueError("reranker_configuration must be a dictionary.")

            reranker_configuration = RerankerConfiguration(
                type=reranker_data["type"],
                model_id=reranker_data.get("model_id"),
                onnx_filename=reranker_data.get("onnx_filename"),
                max_length=reranker_data.get("max_length"),
                candidate_k=reranker_data.get("candidate_k"),
            )

        return cls(
            retriever_type=retriever_type,
            vector_store_type=vector_store_type,
            hybrid_configuration=hybrid_configuration,
            reranker_configuration=reranker_configuration,
        )


@dataclass(frozen=True)
class RetrievalEvaluationLineage:
    """
    Immutable provenance for a retrieval evaluation run.

    This captures evaluation configuration, embedding provenance, and the
    executable retrieval implementation without persisting query text,
    retrieved content, or other evaluation payloads.
    """

    dataset_name: str
    dataset_version: str
    evaluation_policy_name: str | None
    min_recall_at_k: float | None
    min_precision_at_k: float | None
    min_mrr: float | None
    min_ndcg_at_k: float | None
    max_mean_latency_ms: float | None
    min_abstention_accuracy: float | None
    evaluator_k: int
    min_relevance_score: float | None
    embedding_identity: EmbeddingIdentity | None = None
    retrieval_artifact: RetrievalEvaluationArtifact | None = None

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty")

        if not self.dataset_version.strip():
            raise ValueError("dataset_version must not be empty")

        if self.evaluation_policy_name is not None and not self.evaluation_policy_name.strip():
            raise ValueError("evaluation_policy_name must not be empty")

        if self.evaluator_k <= 0:
            raise ValueError("evaluator_k must be greater than zero")

        quality_thresholds = {
            "min_recall_at_k": self.min_recall_at_k,
            "min_precision_at_k": self.min_precision_at_k,
            "min_mrr": self.min_mrr,
            "min_ndcg_at_k": self.min_ndcg_at_k,
            "min_abstention_accuracy": self.min_abstention_accuracy,
        }

        for name, value in quality_thresholds.items():
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        if self.max_mean_latency_ms is not None and self.max_mean_latency_ms < 0.0:
            raise ValueError("max_mean_latency_ms must be non-negative")

        if self.min_relevance_score is not None and not 0.0 <= self.min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between 0.0 and 1.0.")

    def as_dict(self) -> dict[str, object]:
        embedding = self.embedding_identity
        retrieval_artifact = self.retrieval_artifact

        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "evaluation_policy_name": self.evaluation_policy_name,
            "min_recall_at_k": self.min_recall_at_k,
            "min_precision_at_k": self.min_precision_at_k,
            "min_mrr": self.min_mrr,
            "min_ndcg_at_k": self.min_ndcg_at_k,
            "max_mean_latency_ms": self.max_mean_latency_ms,
            "min_abstention_accuracy": self.min_abstention_accuracy,
            "evaluator_k": self.evaluator_k,
            "min_relevance_score": self.min_relevance_score,
            "embedding_requested_provider": (embedding.requested_provider if embedding else None),
            "embedding_requested_model": (embedding.requested_model if embedding else None),
            "embedding_resolved_provider": (embedding.resolved_provider if embedding else None),
            "embedding_resolved_model": (embedding.resolved_model if embedding else None),
            "embedding_dimension": embedding.dimension if embedding else None,
            "retriever_type": (retrieval_artifact.retriever_type if retrieval_artifact else None),
            "vector_store_type": (
                retrieval_artifact.vector_store_type if retrieval_artifact else None
            ),
            "hybrid_configuration": (
                retrieval_artifact.hybrid_configuration.as_dict()
                if retrieval_artifact is not None
                and retrieval_artifact.hybrid_configuration is not None
                else None
            ),
            "reranker_configuration": (
                retrieval_artifact.reranker_configuration.as_dict()
                if retrieval_artifact is not None
                and retrieval_artifact.reranker_configuration is not None
                else None
            ),
        }
