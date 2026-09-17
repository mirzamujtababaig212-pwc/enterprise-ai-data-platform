from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Protocol

from rag.contracts import Retriever
from rag.governance import GovernancePolicy
from rag.models import RetrievalResult

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_PATTERN.findall(text))


class Reranker(Protocol):
    """
    Reorders an existing set of retrieved candidates for a query.

    A reranker does not retrieve additional documents. It only scores and
    orders the candidates supplied to it.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> Sequence[RetrievalResult]: ...


class TokenOverlapReranker:
    """
    Deterministic experimental reranker based on query/document token overlap.

    This implementation is intentionally backend-independent and has no
    knowledge of evaluation labels, benchmark metadata, or relevance grades.

    It is an experimental ranking implementation used to validate the
    reranking abstraction before introducing a learned cross-encoder or
    another model-backed reranker.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> Sequence[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = _tokenize(query)

        scored: list[tuple[RetrievalResult, float, int]] = []

        for original_rank, result in enumerate(candidates):
            document_tokens = _tokenize(result.chunk.content)

            if not query_tokens or not document_tokens:
                overlap_score = 0.0
            else:
                overlap_score = len(query_tokens & document_tokens) / len(query_tokens)

            scored.append((result, overlap_score, original_rank))

        ordered = sorted(
            scored,
            key=lambda item: (
                -item[1],
                item[2],
                item[0].chunk.id,
            ),
        )

        return tuple(
            RetrievalResult(
                chunk=result.chunk,
                score=score,
                embedding_identity=result.embedding_identity,
            )
            for result, score, _ in ordered[:top_k]
        )


class RerankingRetriever:
    """
    Compose an existing Retriever with a post-retrieval Reranker.

    The underlying retriever remains responsible for retrieval, filtering,
    governance, and candidate generation. The reranker only reorders the
    resulting candidate set.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_k: int | None = None,
    ) -> None:
        if candidate_k is not None and candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        self.retriever = retriever
        self.reranker = reranker
        self.candidate_k = candidate_k

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

        candidate_k = self.candidate_k or top_k

        candidates = await self.retriever.retrieve(
            query,
            top_k=max(candidate_k, top_k),
            min_score=min_score,
            metadata_filter=metadata_filter,
            governance_policy=governance_policy,
        )

        return await self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )
