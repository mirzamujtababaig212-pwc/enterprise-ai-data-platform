from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """
    Ground-truth definition for one retrieval evaluation query.

    relevant_chunk_ids contains the chunks considered relevant for binary
    relevance evaluation.

    relevance_grades optionally assigns graded relevance to chunk IDs for
    NDCG evaluation. When omitted, relevant_chunk_ids are treated as
    binary relevance with grade 1.
    """

    query: str
    relevant_chunk_ids: tuple[str, ...]
    relevance_grades: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty.")

        if not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must not be empty.")

        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain duplicates.")

        if any(not chunk_id for chunk_id in self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain empty IDs.")

        if self.relevance_grades is not None:
            if not self.relevance_grades:
                raise ValueError("relevance_grades must not be empty.")

            unknown_ids = set(self.relevance_grades) - set(self.relevant_chunk_ids)
            if unknown_ids:
                raise ValueError(
                    "relevance_grades contains chunk IDs that are not "
                    f"in relevant_chunk_ids: {sorted(unknown_ids)!r}."
                )

            for chunk_id, grade in self.relevance_grades.items():
                if grade < 0:
                    raise ValueError(
                        f"Relevance grade must be non-negative: " f"{chunk_id!r}={grade!r}."
                    )


@dataclass(frozen=True)
class RetrievalQueryResult:
    """
    Retrieved chunk identity and similarity score captured for evaluation.
    """

    chunk_id: str
    score: float


@dataclass(frozen=True)
class RetrievalQueryEvaluation:
    """
    Evaluation result for one retrieval query.

    Failed evaluations retain a structured error type/message while leaving
    retrieval metrics unset.
    """

    query: str
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_results: tuple[RetrievalQueryResult, ...]
    relevant_chunk_ids: tuple[str, ...]
    recall_at_k: float | None
    precision_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """
    Aggregate retrieval evaluation results.
    """

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    evaluated_queries: int
    successful_queries: int
    failed_queries: int
    mean_latency_ms: float
    query_results: tuple[RetrievalQueryEvaluation, ...]

    def as_dict(self) -> dict[str, float]:
        return {
            "retrieval_recall_at_k": self.recall_at_k,
            "retrieval_precision_at_k": self.precision_at_k,
            "retrieval_mrr": self.mrr,
            "retrieval_ndcg_at_k": self.ndcg_at_k,
            "retrieval_evaluated_queries": float(self.evaluated_queries),
            "retrieval_successful_queries": float(self.successful_queries),
            "retrieval_failed_queries": float(self.failed_queries),
            "retrieval_mean_latency_ms": self.mean_latency_ms,
        }
