from __future__ import annotations

from dataclasses import dataclass

from rag.models import EmbeddingIdentity


@dataclass(frozen=True)
class RetrievalEvaluationArtifact:
    """
    Immutable provenance for the retrieval implementation evaluated by a run.

    This describes the executable retrieval components without claiming an
    immutable production index or artifact version.
    """

    retriever_type: str
    vector_store_type: str

    def __post_init__(self) -> None:
        if not self.retriever_type.strip():
            raise ValueError("retriever_type must not be empty")

        if not self.vector_store_type.strip():
            raise ValueError("vector_store_type must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "retriever_type": self.retriever_type,
            "vector_store_type": self.vector_store_type,
        }


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
        }
