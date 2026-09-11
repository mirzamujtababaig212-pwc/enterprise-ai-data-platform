from __future__ import annotations

from dataclasses import dataclass

from rag.models import EmbeddingIdentity


@dataclass(frozen=True)
class RetrievalEvaluationLineage:
    """
    Immutable provenance for a retrieval evaluation run.

    This captures evaluation configuration and embedding provenance without
    persisting query text, retrieved content, or other evaluation payloads.
    """

    dataset_name: str
    dataset_version: str
    evaluation_policy_name: str | None
    evaluator_k: int
    min_relevance_score: float | None
    embedding_identity: EmbeddingIdentity | None = None

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must not be empty")

        if not self.dataset_version.strip():
            raise ValueError("dataset_version must not be empty")

        if self.evaluation_policy_name is not None and not self.evaluation_policy_name.strip():
            raise ValueError("evaluation_policy_name must not be empty")

        if self.evaluator_k <= 0:
            raise ValueError("evaluator_k must be greater than zero")

        if self.min_relevance_score is not None and not 0.0 <= self.min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between 0.0 and 1.0.")

    def as_dict(self) -> dict[str, object]:
        embedding = self.embedding_identity

        return {
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "evaluation_policy_name": self.evaluation_policy_name,
            "evaluator_k": self.evaluator_k,
            "min_relevance_score": self.min_relevance_score,
            "embedding_requested_provider": (embedding.requested_provider if embedding else None),
            "embedding_requested_model": (embedding.requested_model if embedding else None),
            "embedding_resolved_provider": (embedding.resolved_provider if embedding else None),
            "embedding_resolved_model": (embedding.resolved_model if embedding else None),
            "embedding_dimension": embedding.dimension if embedding else None,
        }
