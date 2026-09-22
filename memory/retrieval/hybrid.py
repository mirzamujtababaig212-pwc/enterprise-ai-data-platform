from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from memory.models import MemoryItem, MemoryType
from memory.retrieval.contracts import MemoryRetrievalResult, MemoryRetriever

DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_WEIGHT = 1.0
DEFAULT_LEXICAL_WEIGHT = 0.5


@dataclass(frozen=True)
class HybridRetrievalDiagnostic:
    """Explain how a memory contributed to hybrid RRF ranking."""

    memory_id: str
    semantic_rank: int | None
    lexical_rank: int | None
    semantic_contribution: float
    lexical_contribution: float
    fused_score: float
    final_rank: int


class HybridMemoryRetriever:
    """
    Fuse semantic and lexical memory retrieval using weighted
    Reciprocal Rank Fusion.

    Component retrievers are queried independently. Their raw scores
    are intentionally ignored because lexical and semantic score
    scales are not directly comparable.
    """

    def __init__(
        self,
        semantic_retriever: MemoryRetriever,
        lexical_retriever: MemoryRetriever,
        *,
        candidate_k: int = 10,
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
        *,
        namespace: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> Sequence[MemoryRetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        semantic_results, lexical_results = await asyncio.gather(
            self.semantic_retriever.retrieve(
                query,
                namespace=namespace,
                memory_type=memory_type,
                top_k=self.candidate_k,
            ),
            self.lexical_retriever.retrieve(
                query,
                namespace=namespace,
                memory_type=memory_type,
                top_k=self.candidate_k,
            ),
        )

        fused_scores: defaultdict[str, float] = defaultdict(float)
        item_by_id: dict[str, MemoryItem] = {}
        semantic_ranks: dict[str, int] = {}
        lexical_ranks: dict[str, int] = {}

        for rank, result in enumerate(semantic_results, start=1):
            memory_id = result.item.id
            semantic_ranks[memory_id] = rank
            fused_scores[memory_id] += self.semantic_weight / (self.rrf_k + rank)
            item_by_id.setdefault(memory_id, result.item)

        for rank, result in enumerate(lexical_results, start=1):
            memory_id = result.item.id
            lexical_ranks[memory_id] = rank
            fused_scores[memory_id] += self.lexical_weight / (self.rrf_k + rank)
            item_by_id.setdefault(memory_id, result.item)

        ranked_ids = sorted(
            fused_scores,
            key=lambda memory_id: (
                -fused_scores[memory_id],
                memory_id,
            ),
        )[:top_k]

        return tuple(
            MemoryRetrievalResult(
                item=item_by_id[memory_id],
                retrieval_method="hybrid.rrf",
                rank=rank,
                retrieval_score=fused_scores[memory_id],
                provenance={
                    "semantic_rank": semantic_ranks.get(memory_id),
                    "lexical_rank": lexical_ranks.get(memory_id),
                    "semantic_contribution": (
                        self.semantic_weight / (self.rrf_k + semantic_ranks[memory_id])
                        if memory_id in semantic_ranks
                        else 0.0
                    ),
                    "lexical_contribution": (
                        self.lexical_weight / (self.rrf_k + lexical_ranks[memory_id])
                        if memory_id in lexical_ranks
                        else 0.0
                    ),
                },
            )
            for rank, memory_id in enumerate(ranked_ids, start=1)
        )

    async def diagnose(
        self,
        query: str,
        *,
        namespace: str,
        memory_type: MemoryType | None = None,
        top_k: int = 5,
    ) -> Sequence[HybridRetrievalDiagnostic]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if not namespace.strip():
            raise ValueError("Memory namespace must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        semantic_results, lexical_results = await asyncio.gather(
            self.semantic_retriever.retrieve(
                query,
                namespace=namespace,
                memory_type=memory_type,
                top_k=self.candidate_k,
            ),
            self.lexical_retriever.retrieve(
                query,
                namespace=namespace,
                memory_type=memory_type,
                top_k=self.candidate_k,
            ),
        )

        semantic_ranks = {
            result.item.id: rank for rank, result in enumerate(semantic_results, start=1)
        }
        lexical_ranks = {
            result.item.id: rank for rank, result in enumerate(lexical_results, start=1)
        }

        item_ids = set(semantic_ranks) | set(lexical_ranks)
        fused_scores: dict[str, float] = {}

        for memory_id in item_ids:
            semantic_rank = semantic_ranks.get(memory_id)
            lexical_rank = lexical_ranks.get(memory_id)

            semantic_contribution = (
                self.semantic_weight / (self.rrf_k + semantic_rank)
                if semantic_rank is not None
                else 0.0
            )
            lexical_contribution = (
                self.lexical_weight / (self.rrf_k + lexical_rank)
                if lexical_rank is not None
                else 0.0
            )

            fused_scores[memory_id] = semantic_contribution + lexical_contribution

        ranked_ids = sorted(
            fused_scores,
            key=lambda memory_id: (
                -fused_scores[memory_id],
                memory_id,
            ),
        )[:top_k]

        return tuple(
            HybridRetrievalDiagnostic(
                memory_id=memory_id,
                semantic_rank=semantic_ranks.get(memory_id),
                lexical_rank=lexical_ranks.get(memory_id),
                semantic_contribution=(
                    self.semantic_weight / (self.rrf_k + semantic_ranks[memory_id])
                    if memory_id in semantic_ranks
                    else 0.0
                ),
                lexical_contribution=(
                    self.lexical_weight / (self.rrf_k + lexical_ranks[memory_id])
                    if memory_id in lexical_ranks
                    else 0.0
                ),
                fused_score=fused_scores[memory_id],
                final_rank=rank,
            )
            for rank, memory_id in enumerate(ranked_ids, start=1)
        )
