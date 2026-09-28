from __future__ import annotations

import time
from collections.abc import Sequence

from rag.contracts import Retriever
from rag.models import EmbeddingIdentity
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
        min_relevance_score: float | None = None,
        embedding_identity: EmbeddingIdentity | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if min_relevance_score is not None and not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between 0.0 and 1.0.")

        self.retriever = retriever
        self.k = k
        self.min_relevance_score = min_relevance_score
        self.embedding_identity = embedding_identity

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
                    min_score=self.min_relevance_score,
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
                        expect_abstention=case.expect_abstention,
                        abstention_correct=None,
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
                    retrieval_score=(
                        getattr(result, "retrieval_score", None)
                        if getattr(result, "retrieval_score", None) is not None
                        else result.score
                    ),
                    reranker_score=getattr(result, "reranker_score", None),
                )
                for result in retrieved_results[: self.k]
            )

            retrieved_ids = tuple(result.chunk_id for result in retrieved_results)

            if case.expect_abstention:
                recall = None
                precision = None
                reciprocal = None
                ndcg = None
                abstention_correct = not retrieved_ids
            else:
                if case.relevance_grades is None:
                    relevance_grades = {chunk_id: 1.0 for chunk_id in case.relevant_chunk_ids}
                else:
                    relevance_grades = dict(case.relevance_grades)

                recall = recall_at_k(
                    retrieved_ids,
                    case.relevant_chunk_ids,
                    self.k,
                )
                precision = precision_at_k(
                    retrieved_ids,
                    case.relevant_chunk_ids,
                    self.k,
                )
                reciprocal = reciprocal_rank(
                    retrieved_ids,
                    case.relevant_chunk_ids,
                )
                ndcg = ndcg_at_k(
                    retrieved_ids,
                    relevance_grades,
                    self.k,
                )
                abstention_correct = None

            query_results.append(
                RetrievalQueryEvaluation(
                    query=case.query,
                    retrieved_chunk_ids=retrieved_ids,
                    retrieved_results=retrieved_results,
                    relevant_chunk_ids=case.relevant_chunk_ids,
                    recall_at_k=recall,
                    precision_at_k=precision,
                    reciprocal_rank=reciprocal,
                    ndcg_at_k=ndcg,
                    latency_ms=latency_ms,
                    expect_abstention=case.expect_abstention,
                    abstention_correct=abstention_correct,
                )
            )

        successful_results = [result for result in query_results if result.error_type is None]
        answerable_results = [
            result for result in successful_results if not result.expect_abstention
        ]
        abstention_results = [result for result in successful_results if result.expect_abstention]

        successful_queries = len(successful_results)
        answerable_queries = len(answerable_results)
        abstention_evaluated_queries = len(abstention_results)
        evaluated_queries = len(cases)

        abstention_accuracy = (
            sum(result.abstention_correct is True for result in abstention_results)
            / abstention_evaluated_queries
            if abstention_evaluated_queries
            else 0.0
        )

        mean_latency_ms = (
            sum(result.latency_ms for result in query_results) / len(query_results)
            if query_results
            else 0.0
        )

        retrieval_scores = [
            retrieved.retrieval_score
            for query_result in successful_results
            for retrieved in query_result.retrieved_results
            if retrieved.retrieval_score is not None
        ]
        reranker_scores = [
            retrieved.reranker_score
            for query_result in successful_results
            for retrieved in query_result.retrieved_results
            if retrieved.reranker_score is not None
        ]

        retrieval_score_min = min(retrieval_scores) if retrieval_scores else None
        retrieval_score_max = max(retrieval_scores) if retrieval_scores else None
        retrieval_score_avg = (
            sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else None
        )

        reranker_score_min = min(reranker_scores) if reranker_scores else None
        reranker_score_max = max(reranker_scores) if reranker_scores else None
        reranker_score_avg = (
            sum(reranker_scores) / len(reranker_scores) if reranker_scores else None
        )

        if answerable_queries == 0:
            return RetrievalEvaluationResult(
                recall_at_k=0.0,
                precision_at_k=0.0,
                mrr=0.0,
                ndcg_at_k=0.0,
                evaluated_queries=evaluated_queries,
                successful_queries=successful_queries,
                failed_queries=failed_queries,
                mean_latency_ms=mean_latency_ms,
                abstention_accuracy=abstention_accuracy,
                abstention_evaluated_queries=abstention_evaluated_queries,
                query_results=tuple(query_results),
                retrieval_score_min=retrieval_score_min,
                retrieval_score_max=retrieval_score_max,
                retrieval_score_avg=retrieval_score_avg,
                reranker_score_min=reranker_score_min,
                reranker_score_max=reranker_score_max,
                reranker_score_avg=reranker_score_avg,
            )

        return RetrievalEvaluationResult(
            recall_at_k=sum(
                result.recall_at_k
                for result in answerable_results
                if result.recall_at_k is not None
            )
            / answerable_queries,
            precision_at_k=sum(
                result.precision_at_k
                for result in answerable_results
                if result.precision_at_k is not None
            )
            / answerable_queries,
            mrr=sum(
                result.reciprocal_rank
                for result in answerable_results
                if result.reciprocal_rank is not None
            )
            / answerable_queries,
            ndcg_at_k=sum(
                result.ndcg_at_k for result in answerable_results if result.ndcg_at_k is not None
            )
            / answerable_queries,
            evaluated_queries=evaluated_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            mean_latency_ms=mean_latency_ms,
            abstention_accuracy=abstention_accuracy,
            abstention_evaluated_queries=abstention_evaluated_queries,
            query_results=tuple(query_results),
            retrieval_score_min=retrieval_score_min,
            retrieval_score_max=retrieval_score_max,
            retrieval_score_avg=retrieval_score_avg,
            reranker_score_min=reranker_score_min,
            reranker_score_max=reranker_score_max,
            reranker_score_avg=reranker_score_avg,
        )
