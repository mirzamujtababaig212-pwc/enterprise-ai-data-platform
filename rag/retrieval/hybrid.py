from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.contracts import Retriever
from rag.governance import GovernancePolicy
from rag.models import RetrievalResult

DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_LEXICAL_WEIGHT = 0.5


@dataclass(frozen=True)
class HybridRetrievalDiagnostic:
    """Explain how a chunk contributed to hybrid RRF ranking."""

    chunk_id: str
    semantic_rank: int | None
    lexical_rank: int | None
    semantic_contribution: float
    lexical_contribution: float
    fused_score: float
    final_rank: int


class HybridRetriever:
    """
    Fuse semantic and lexical retrieval using weighted Reciprocal Rank Fusion.

    Component retrievers are queried independently and their ranked results
    are combined by rank rather than raw score, because semantic and lexical
    score scales are not directly comparable.

    semantic_weight and lexical_weight control each retriever's contribution
    to the fused RRF score. Component min_score filtering is applied before
    fusion; min_score is not a threshold on the final fused RRF score.
    """

    def __init__(
        self,
        semantic_retriever: Retriever,
        lexical_retriever: Retriever,
        *,
        candidate_k: int = 5,
        rrf_k: int = DEFAULT_RRF_K,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if semantic_weight < 0:
            raise ValueError("semantic_weight must be greater than or equal to zero.")

        if lexical_weight < 0:
            raise ValueError("lexical_weight must be greater than or equal to zero.")

        if semantic_weight == 0 and lexical_weight == 0:
            raise ValueError("At least one retrieval weight must be greater than zero.")

        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

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

        fused_results, _ = self._fuse_results(
            semantic_results,
            lexical_results,
        )

        return fused_results[:top_k]

    async def diagnose(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        governance_policy: GovernancePolicy | None = None,
    ) -> Sequence[HybridRetrievalDiagnostic]:
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

        _, diagnostics = self._fuse_results(
            semantic_results,
            lexical_results,
        )

        return diagnostics[:top_k]

    def _fuse_results(
        self,
        semantic_results: Sequence[RetrievalResult],
        lexical_results: Sequence[RetrievalResult],
    ) -> tuple[
        list[RetrievalResult],
        list[HybridRetrievalDiagnostic],
    ]:
        semantic_ranks = {
            result.chunk.id: rank for rank, result in enumerate(semantic_results, start=1)
        }
        lexical_ranks = {
            result.chunk.id: rank for rank, result in enumerate(lexical_results, start=1)
        }

        fused_scores: defaultdict[str, float] = defaultdict(float)
        result_by_chunk_id: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(semantic_results, start=1):
            fused_scores[result.chunk.id] += self.semantic_weight / (self.rrf_k + rank)
            result_by_chunk_id.setdefault(result.chunk.id, result)

        for rank, result in enumerate(lexical_results, start=1):
            fused_scores[result.chunk.id] += self.lexical_weight / (self.rrf_k + rank)
            result_by_chunk_id.setdefault(result.chunk.id, result)

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )

        fused_results = [
            RetrievalResult(
                chunk=result_by_chunk_id[chunk_id].chunk,
                score=fused_scores[chunk_id],
                embedding_identity=result_by_chunk_id[chunk_id].embedding_identity,
            )
            for chunk_id in ranked_chunk_ids
        ]

        diagnostics = [
            HybridRetrievalDiagnostic(
                chunk_id=chunk_id,
                semantic_rank=semantic_ranks.get(chunk_id),
                lexical_rank=lexical_ranks.get(chunk_id),
                semantic_contribution=(
                    self.semantic_weight / (self.rrf_k + semantic_ranks[chunk_id])
                    if chunk_id in semantic_ranks
                    else 0.0
                ),
                lexical_contribution=(
                    self.lexical_weight / (self.rrf_k + lexical_ranks[chunk_id])
                    if chunk_id in lexical_ranks
                    else 0.0
                ),
                fused_score=fused_scores[chunk_id],
                final_rank=rank,
            )
            for rank, chunk_id in enumerate(ranked_chunk_ids, start=1)
        ]

        return fused_results, diagnostics
