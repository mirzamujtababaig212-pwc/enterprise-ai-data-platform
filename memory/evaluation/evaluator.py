from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

from .models import (
    MemoryRetrievalEvaluationCase,
    MemoryRetrievalEvaluationResult,
)


@dataclass(frozen=True)
class MemoryRetrievalQueryEvaluation:
    query: str
    retrieved_memory_ids: tuple[str, ...]
    relevant_memory_ids: tuple[str, ...]
    recall_at_k: float | None
    precision_at_k: float | None
    reciprocal_rank: float | None
    ndcg_at_k: float | None
    latency_ms: float
    error_type: str | None = None
    error_message: str | None = None


class MemoryRetrievalEvaluator:
    """
    Evaluates memory retrieval quality against labeled memory cases.

    The evaluator is memory-domain specific while reusing the generic
    retrieval metric implementations.
    """

    def __init__(
        self,
        retriever,
        *,
        k: int = 5,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        self.retriever = retriever
        self.k = k

    async def evaluate(
        self,
        cases: Sequence[MemoryRetrievalEvaluationCase],
    ) -> MemoryRetrievalEvaluationResult:
        if not cases:
            raise ValueError("cases must not be empty.")

        query_results: list[MemoryRetrievalQueryEvaluation] = []
        failed_queries = 0

        for case in cases:
            start_time = time.perf_counter()

            try:
                retrieved = await self.retriever.retrieve(
                    case.query,
                    namespace=case.namespace,
                    memory_type=case.memory_type,
                    top_k=self.k,
                )
            except Exception as exc:
                failed_queries += 1
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                query_results.append(
                    MemoryRetrievalQueryEvaluation(
                        query=case.query,
                        retrieved_memory_ids=(),
                        relevant_memory_ids=case.relevant_memory_ids,
                        recall_at_k=None,
                        precision_at_k=None,
                        reciprocal_rank=None,
                        ndcg_at_k=None,
                        latency_ms=latency_ms,
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
                )
                continue

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            retrieved_ids = tuple(result.item.id for result in retrieved[: self.k])

            if case.relevance_grades is None:
                relevance_grades = {memory_id: 1.0 for memory_id in case.relevant_memory_ids}
            else:
                relevance_grades = dict(case.relevance_grades)

            query_results.append(
                MemoryRetrievalQueryEvaluation(
                    query=case.query,
                    retrieved_memory_ids=retrieved_ids,
                    relevant_memory_ids=case.relevant_memory_ids,
                    recall_at_k=recall_at_k(
                        retrieved_ids,
                        case.relevant_memory_ids,
                        self.k,
                    ),
                    precision_at_k=precision_at_k(
                        retrieved_ids,
                        case.relevant_memory_ids,
                        self.k,
                    ),
                    reciprocal_rank=reciprocal_rank(
                        retrieved_ids,
                        case.relevant_memory_ids,
                    ),
                    ndcg_at_k=ndcg_at_k(
                        retrieved_ids,
                        relevance_grades,
                        self.k,
                    ),
                    latency_ms=latency_ms,
                )
            )

        successful_results = [result for result in query_results if result.error_type is None]

        successful_queries = len(successful_results)
        evaluated_queries = len(cases)

        mean_latency_ms = sum(result.latency_ms for result in query_results) / len(query_results)

        if successful_queries == 0:
            return MemoryRetrievalEvaluationResult(
                recall_at_k=0.0,
                precision_at_k=0.0,
                mrr=0.0,
                ndcg_at_k=0.0,
                evaluated_queries=evaluated_queries,
                successful_queries=0,
                failed_queries=failed_queries,
                mean_latency_ms=mean_latency_ms,
                query_results=tuple(query_results),
            )

        return MemoryRetrievalEvaluationResult(
            recall_at_k=sum(
                result.recall_at_k
                for result in successful_results
                if result.recall_at_k is not None
            )
            / successful_queries,
            precision_at_k=sum(
                result.precision_at_k
                for result in successful_results
                if result.precision_at_k is not None
            )
            / successful_queries,
            mrr=sum(
                result.reciprocal_rank
                for result in successful_results
                if result.reciprocal_rank is not None
            )
            / successful_queries,
            ndcg_at_k=sum(
                result.ndcg_at_k for result in successful_results if result.ndcg_at_k is not None
            )
            / successful_queries,
            evaluated_queries=evaluated_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            mean_latency_ms=mean_latency_ms,
            query_results=tuple(query_results),
        )
