from __future__ import annotations

from collections.abc import Sequence

from memory.models import MemoryItem, MemoryType
from memory.retrieval.contracts import MemoryRetriever
from memory.retrieval.reranker import MemoryReranker


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
    ) -> Sequence[MemoryItem]:
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

        return await self.reranker.rerank(
            query,
            candidates,
            top_k=top_k,
        )
