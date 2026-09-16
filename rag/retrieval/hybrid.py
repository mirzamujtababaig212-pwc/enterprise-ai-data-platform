from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence

from rag.contracts import Retriever
from rag.governance import GovernancePolicy
from rag.models import RetrievalResult

DEFAULT_RRF_K = 60


class HybridRetriever:
    """
    Fuse semantic and lexical retrieval using Reciprocal Rank Fusion.

    Component retrievers are queried independently and their ranked results
    are combined by rank rather than raw score, because semantic and lexical
    score scales are not directly comparable.
    """

    def __init__(
        self,
        semantic_retriever: Retriever,
        lexical_retriever: Retriever,
        *,
        candidate_k: int = 5,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

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

        semantic_results, lexical_results = await asyncio.gather(
            self.semantic_retriever.retrieve(
                query,
                top_k=self.candidate_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
                governance_policy=governance_policy,
            ),
            self.lexical_retriever.retrieve(
                query,
                top_k=self.candidate_k,
                min_score=min_score,
                metadata_filter=metadata_filter,
                governance_policy=governance_policy,
            ),
        )

        fused_scores: defaultdict[str, float] = defaultdict(float)
        result_by_chunk_id: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(semantic_results, start=1):
            fused_scores[result.chunk.id] += 1.0 / (self.rrf_k + rank)
            result_by_chunk_id.setdefault(result.chunk.id, result)

        for rank, result in enumerate(lexical_results, start=1):
            fused_scores[result.chunk.id] += 1.0 / (self.rrf_k + rank)
            result_by_chunk_id.setdefault(result.chunk.id, result)

        fused_results = [
            RetrievalResult(
                chunk=result_by_chunk_id[chunk_id].chunk,
                score=fused_scores[chunk_id],
                embedding_identity=result_by_chunk_id[chunk_id].embedding_identity,
            )
            for chunk_id in fused_scores
        ]

        return sorted(
            fused_results,
            key=lambda result: (-result.score, result.chunk.id),
        )[:top_k]
