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

    expect_abstention marks a query as intentionally unanswerable. Such
    cases must not define relevant chunks and are evaluated based on whether
    retrieval returns no results.
    """

    query: str
    relevant_chunk_ids: tuple[str, ...]
    relevance_grades: Mapping[str, float] | None = None
    expect_abstention: bool = False

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty.")

        if self.expect_abstention and self.relevant_chunk_ids:
            raise ValueError("expect_abstention cases must not define relevant_chunk_ids.")

        if not self.expect_abstention and not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must not be empty.")

        if len(set(self.relevant_chunk_ids)) != len(self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain duplicates.")

        if any(not chunk_id for chunk_id in self.relevant_chunk_ids):
            raise ValueError("relevant_chunk_ids must not contain empty IDs.")

        if self.expect_abstention and self.relevance_grades is not None:
            raise ValueError("expect_abstention cases must not define relevance_grades.")

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
    Retrieved chunk identity and ranking scores captured for evaluation.

    ``score`` remains the final ranking score for backward compatibility.
    ``retrieval_score`` preserves the original retriever score, while
    ``reranker_score`` records a reranker's score when one was used.
    """

    chunk_id: str
    score: float
    retrieval_score: float | None = None
    reranker_score: float | None = None


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
    expect_abstention: bool = False
    abstention_correct: bool | None = None
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
    abstention_accuracy: float = 0.0
    abstention_evaluated_queries: int = 0
    retrieval_score_min: float | None = None
    retrieval_score_max: float | None = None
    retrieval_score_avg: float | None = None
    reranker_score_min: float | None = None
    reranker_score_max: float | None = None
    reranker_score_avg: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "retrieval_recall_at_k": self.recall_at_k,
            "retrieval_precision_at_k": self.precision_at_k,
            "retrieval_mrr": self.mrr,
            "retrieval_ndcg_at_k": self.ndcg_at_k,
            "retrieval_evaluated_queries": float(self.evaluated_queries),
            "retrieval_successful_queries": float(self.successful_queries),
            "retrieval_failed_queries": float(self.failed_queries),
            "retrieval_mean_latency_ms": self.mean_latency_ms,
            "retrieval_abstention_accuracy": self.abstention_accuracy,
            "retrieval_abstention_evaluated_queries": float(self.abstention_evaluated_queries),
            "retrieval_score_min": self.retrieval_score_min,
            "retrieval_score_max": self.retrieval_score_max,
            "retrieval_score_avg": self.retrieval_score_avg,
            "reranker_score_min": self.reranker_score_min,
            "reranker_score_max": self.reranker_score_max,
            "reranker_score_avg": self.reranker_score_avg,
        }
