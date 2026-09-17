import pytest

from rag.governance import GovernancePolicy
from rag.models import DocumentChunk, RetrievalResult
from rag.retrieval import RerankingRetriever


def _result(chunk_id: str, content: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            content=content,
        ),
        score=score,
    )


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: dict[str, object] | None = None,
        governance_policy: GovernancePolicy | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
                "metadata_filter": metadata_filter,
                "governance_policy": governance_policy,
            }
        )
        return self.results[:top_k]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append(
            {
                "query": query,
                "candidates": candidates,
                "top_k": top_k,
            }
        )
        return list(reversed(candidates))[:top_k]


@pytest.mark.asyncio
async def test_reranking_retriever_composes_retriever_and_reranker():
    retriever = FakeRetriever(
        [
            _result("a", "vehicle", 0.9),
            _result("b", "battery", 0.8),
            _result("c", "motor", 0.7),
        ]
    )
    reranker = FakeReranker()

    service = RerankingRetriever(
        retriever,
        reranker,
        candidate_k=3,
    )

    results = await service.retrieve(
        "vehicle battery",
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["c", "b"]

    assert retriever.calls[0]["top_k"] == 3
    assert reranker.calls[0]["top_k"] == 2
    assert reranker.calls[0]["query"] == "vehicle battery"


@pytest.mark.asyncio
async def test_reranking_retriever_uses_top_k_when_candidate_k_not_set():
    retriever = FakeRetriever(
        [
            _result("a", "vehicle", 0.9),
            _result("b", "battery", 0.8),
        ]
    )
    reranker = FakeReranker()

    service = RerankingRetriever(retriever, reranker)

    await service.retrieve("vehicle", top_k=2)

    assert retriever.calls[0]["top_k"] == 2


@pytest.mark.asyncio
async def test_reranking_retriever_propagates_filters_and_governance():
    retriever = FakeRetriever([_result("a", "vehicle", 0.9)])
    reranker = FakeReranker()

    policy = GovernancePolicy(
        required_metadata={"classification": "internal"},
    )

    service = RerankingRetriever(retriever, reranker)

    await service.retrieve(
        "vehicle",
        metadata_filter={"source": "architecture.md"},
        governance_policy=policy,
    )

    call = retriever.calls[0]

    assert call["metadata_filter"] == {"source": "architecture.md"}
    assert call["governance_policy"] is policy


@pytest.mark.asyncio
async def test_reranking_retriever_rejects_invalid_arguments():
    retriever = FakeRetriever([])
    reranker = FakeReranker()

    with pytest.raises(ValueError, match="candidate_k"):
        RerankingRetriever(
            retriever,
            reranker,
            candidate_k=0,
        )

    service = RerankingRetriever(retriever, reranker)

    with pytest.raises(ValueError, match="empty"):
        await service.retrieve("")

    with pytest.raises(ValueError, match="top_k"):
        await service.retrieve("vehicle", top_k=0)

    with pytest.raises(ValueError, match="between -1.0 and 1.0"):
        await service.retrieve("vehicle", min_score=1.1)
