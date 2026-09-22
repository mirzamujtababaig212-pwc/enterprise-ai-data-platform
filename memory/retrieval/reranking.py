from __future__ import annotations

from collections.abc import Sequence

from memory.models import MemoryType
from memory.retrieval.contracts import MemoryRetrievalResult, MemoryRetriever
from memory.retrieval.reranker import MemoryReranker, MemoryRerankResult


class RerankingMemoryRetriever:
    """
    Compose a memory retriever with a post-retrieval memory reranker.

    The underlying retriever remains responsible for candidate generation,
    namespace filtering, memory-type filtering, and expiration handling.
    The reranker only reorders the resulting candidate set.
    """

    def __init__(
        self,
        retriever: MemoryRetriever,
        reranker: MemoryReranker,
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

        candidate_k = self.candidate_k or top_k

        candidates = await self.retriever.retrieve(
            query,
            namespace=namespace,
            memory_type=memory_type,
            top_k=max(candidate_k, top_k),
        )

        candidate_items = tuple(result.item for result in candidates)

        rerank_with_scores = getattr(self.reranker, "rerank_with_scores", None)

        if rerank_with_scores is not None:
            reranked = await rerank_with_scores(
                query,
                candidate_items,
                top_k=top_k,
            )
        else:
            reranked_items = await self.reranker.rerank(
                query,
                candidate_items,
                top_k=top_k,
            )
            reranked = tuple(
                MemoryRerankResult(
                    item=item,
                    score=float("nan"),
                    original_rank=index,
                )
                for index, item in enumerate(reranked_items, start=1)
            )

        candidate_by_id = {result.item.id: result for result in candidates}

        results: list[MemoryRetrievalResult] = []

        for final_rank, rerank_result in enumerate(reranked, start=1):
            original = candidate_by_id.get(rerank_result.item.id)

            if original is None:
                raise ValueError(
                    "Memory reranker returned a candidate that was not supplied "
                    "by the underlying retriever."
                )

            provenance = dict(original.provenance)
            provenance["original_retrieval_rank"] = original.rank
            provenance["reranker_original_rank"] = rerank_result.original_rank

            results.append(
                MemoryRetrievalResult(
                    item=original.item,
                    retrieval_method=f"{original.retrieval_method}+rerank",
                    rank=final_rank,
                    retrieval_score=original.retrieval_score,
                    reranker_score=rerank_result.score,
                    provenance=provenance,
                )
            )

        return tuple(results)
