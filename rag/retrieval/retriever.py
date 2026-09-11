from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from opentelemetry import trace

from rag.contracts import EmbeddingService, VectorStore
from rag.governance import GovernancePolicy
from rag.models import RetrievalResult
from rag.observability import (
    RAG_RETRIEVAL_DURATION_SECONDS,
    RAG_RETRIEVAL_EMPTY_RESULTS_TOTAL,
    RAG_RETRIEVAL_ERRORS_TOTAL,
    RAG_RETRIEVAL_REQUESTS_TOTAL,
    RAG_RETRIEVAL_TRACER,
)


class SemanticRetriever:
    """
    Query-to-vector retrieval service.

    The retriever knows about embedding, vector-store, governance,
    and retrieval observability contracts, but knows nothing about
    a specific embedding provider, database, or enterprise data source.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        *,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self._tracer = tracer or RAG_RETRIEVAL_TRACER

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        governance_policy: GovernancePolicy | None = None,
    ) -> Sequence[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if min_score is not None and not -1.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between -1.0 and 1.0.")

        effective_metadata_filter = self._build_metadata_filter(
            metadata_filter=metadata_filter,
            governance_policy=governance_policy,
        )

        start_time = time.perf_counter()

        RAG_RETRIEVAL_REQUESTS_TOTAL.inc()

        with self._tracer.start_as_current_span("rag.retrieval") as span:
            span.set_attribute("rag.retrieval.top_k", top_k)

            if min_score is not None:
                span.set_attribute(
                    "rag.retrieval.min_score",
                    min_score,
                )

            span.set_attribute(
                "rag.retrieval.metadata_filter_applied",
                metadata_filter is not None,
            )

            span.set_attribute(
                "rag.retrieval.governance_policy_applied",
                governance_policy is not None,
            )

            try:
                embedding = await self.embedding_service.embed(query)

                results = await self.vector_store.search(
                    embedding,
                    top_k=top_k,
                    metadata_filter=effective_metadata_filter,
                )

                candidate_count = len(results)

                filtered_results = [
                    result for result in results if min_score is None or result.score >= min_score
                ]

                ordered_results = sorted(
                    filtered_results,
                    key=lambda result: (-result.score, result.chunk.id),
                )

                returned_count = len(ordered_results)

                span.set_attribute(
                    "rag.retrieval.candidate_count",
                    candidate_count,
                )
                span.set_attribute(
                    "rag.retrieval.returned_count",
                    returned_count,
                )

                if ordered_results:
                    scores = [result.score for result in ordered_results]

                    span.set_attribute(
                        "rag.retrieval.score_min",
                        min(scores),
                    )
                    span.set_attribute(
                        "rag.retrieval.score_max",
                        max(scores),
                    )
                    span.set_attribute(
                        "rag.retrieval.score_avg",
                        sum(scores) / len(scores),
                    )
                else:
                    RAG_RETRIEVAL_EMPTY_RESULTS_TOTAL.inc()

                return ordered_results

            except Exception:
                RAG_RETRIEVAL_ERRORS_TOTAL.inc()
                span.set_status(trace.StatusCode.ERROR)
                raise

            finally:
                duration = time.perf_counter() - start_time
                RAG_RETRIEVAL_DURATION_SECONDS.observe(duration)

    @staticmethod
    def _build_metadata_filter(
        *,
        metadata_filter: Mapping[str, object] | None,
        governance_policy: GovernancePolicy | None,
    ) -> dict[str, object] | None:
        if metadata_filter is None and governance_policy is None:
            return None

        effective_filter = dict(metadata_filter or {})

        if governance_policy is None:
            return effective_filter

        policy_filter = governance_policy.to_metadata_filter()

        for key, policy_value in policy_filter.items():
            if key in effective_filter and effective_filter[key] != policy_value:
                raise ValueError(
                    f"Metadata filter conflicts with governance policy for key '{key}'."
                )

            effective_filter[key] = policy_value

        return effective_filter
