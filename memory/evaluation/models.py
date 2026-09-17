from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from memory.models import MemoryType


@dataclass(frozen=True)
class MemoryRetrievalEvaluationCase:
    """
    Ground-truth definition for one memory retrieval evaluation query.
    """

    query: str
    namespace: str
    relevant_memory_ids: tuple[str, ...]
    memory_type: MemoryType | None = None
    relevance_grades: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty.")

        if not self.namespace.strip():
            raise ValueError("namespace must not be empty.")

        if not self.relevant_memory_ids:
            raise ValueError("relevant_memory_ids must not be empty.")

        if len(set(self.relevant_memory_ids)) != len(self.relevant_memory_ids):
            raise ValueError("relevant_memory_ids must not contain duplicates.")

        if any(not memory_id for memory_id in self.relevant_memory_ids):
            raise ValueError("relevant_memory_ids must not contain empty IDs.")

        if self.relevance_grades is not None:
            if not self.relevance_grades:
                raise ValueError("relevance_grades must not be empty.")

            unknown_ids = set(self.relevance_grades) - set(self.relevant_memory_ids)
            if unknown_ids:
                raise ValueError(
                    "relevance_grades contains memory IDs that are not "
                    f"in relevant_memory_ids: {sorted(unknown_ids)!r}."
                )

            for memory_id, grade in self.relevance_grades.items():
                if grade < 0:
                    raise ValueError(
                        f"Relevance grade must be non-negative: " f"{memory_id!r}={grade!r}."
                    )


@dataclass(frozen=True)
class MemoryRetrievalEvaluationResult:
    """
    Aggregate memory retrieval evaluation results.
    """

    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    evaluated_queries: int
    successful_queries: int
    failed_queries: int
    mean_latency_ms: float
    query_results: tuple[object, ...] = ()

    def as_dict(self) -> dict[str, float]:
        return {
            "memory_retrieval_recall_at_k": self.recall_at_k,
            "memory_retrieval_precision_at_k": self.precision_at_k,
            "memory_retrieval_mrr": self.mrr,
            "memory_retrieval_ndcg_at_k": self.ndcg_at_k,
            "memory_retrieval_evaluated_queries": float(self.evaluated_queries),
            "memory_retrieval_successful_queries": float(self.successful_queries),
            "memory_retrieval_failed_queries": float(self.failed_queries),
            "memory_retrieval_mean_latency_ms": self.mean_latency_ms,
        }
