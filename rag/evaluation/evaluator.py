from __future__ import annotations

import time
from collections.abc import Sequence

from rag.contracts import Retriever
from rag.evaluation.metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.evaluation.models import (
    RetrievalEvaluationCase,
    RetrievalEvaluationResult,
    RetrievalQueryEvaluation,
    RetrievalQueryResult,
)


class RetrievalEvaluator:
    """
    Evaluates retrieval quality against labeled query/chunk cases.

    This evaluator is intentionally async-native because the canonical
    Retriever contract is asynchronous.
    """

    def __init__(
        self,
        retriever: Retriever,
        *,
        k: int = 5,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        self.retriever = retriever
        self.k = k

    async def evaluate(
        self,
        cases: Sequence[RetrievalEvaluationCase],
    ) -> RetrievalEvaluationResult:
        if not cases:
            raise ValueError("cases must not be empty.")

        query_results: list[RetrievalQueryEvaluation] = []
        failed_queries = 0

        for case in cases:
            start_time = time.perf_counter()

            try:
                retrieved_results = await self.retriever.retrieve(
                    query=case.query,
                    top_k=self.k,
                )
            except Exception as exc:
                failed_queries += 1
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                query_results.append(
                    RetrievalQueryEvaluation(
                        query=case.query,
                        retrieved_chunk_ids=(),
                        retrieved_results=(),
                        relevant_chunk_ids=case.relevant_chunk_ids,
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

            retrieved_results = tuple(
                RetrievalQueryResult(
                    chunk_id=result.chunk.id,
                    score=result.score,
                )
                for result in retrieved_results[: self.k]
            )

            retrieved_ids = tuple(result.chunk_id for result in retrieved_results)

            if case.relevance_grades is None:
                relevance_grades = {chunk_id: 1.0 for chunk_id in case.relevant_chunk_ids}
            else:
                relevance_grades = dict(case.relevance_grades)

            query_results.append(
                RetrievalQueryEvaluation(
                    query=case.query,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieved_results=retrieved_results,
                    relevant_chunk_ids=case.relevant_chunk_ids,
                    recall_at_k=recall_at_k(
                        retrieved_ids,
                        case.relevant_chunk_ids,
                        self.k,
                    ),
                    precision_at_k=precision_at_k(
                        retrieved_ids,
                        case.relevant_chunk_ids,
                        self.k,
                    ),
                    reciprocal_rank=reciprocal_rank(
                        retrieved_ids,
                        case.relevant_chunk_ids,
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

        if successful_queries == 0:
            return RetrievalEvaluationResult(
                recall_at_k=0.0,
                precision_at_k=0.0,
                mrr=0.0,
                ndcg_at_k=0.0,
                evaluated_queries=evaluated_queries,
                successful_queries=0,
                failed_queries=failed_queries,
                mean_latency_ms=(
                    sum(result.latency_ms for result in query_results) / len(query_results)
                    if query_results
                    else 0.0
                ),
                query_results=tuple(query_results),
            )

        return RetrievalEvaluationResult(
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
            mean_latency_ms=(
                sum(result.latency_ms for result in query_results) / len(query_results)
            ),
            query_results=tuple(query_results),
        )
