import pytest

from rag.governance import GovernancePolicy
from rag.models import DocumentChunk, RetrievalResult
from rag.retrieval import HybridRetriever


def _result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            content=f"content for {chunk_id}",
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


@pytest.mark.asyncio
async def test_hybrid_retriever_fuses_overlapping_results_with_rrf():
    semantic = FakeRetriever(
        [
            _result("shared", 0.95),
            _result("semantic-only", 0.80),
            _result("semantic-third", 0.70),
        ]
    )
    lexical = FakeRetriever(
        [
            _result("lexical-only", 1.0),
            _result("shared", 0.90),
            _result("lexical-third", 0.60),
        ]
    )

    retriever = HybridRetriever(
        semantic,
        lexical,
        candidate_k=3,
        rrf_k=60,
    )

    results = await retriever.retrieve(
        "vehicle battery",
        top_k=4,
    )

    assert [result.chunk.id for result in results] == [
        "shared",
        "lexical-only",
        "semantic-only",
        "lexical-third",
    ]

    expected_shared = (1 / 61) + (1 / 62)
    assert results[0].score == pytest.approx(expected_shared)

    assert len(results) == 4


@pytest.mark.asyncio
async def test_hybrid_retriever_uses_candidate_k_for_both_retrievers():
    semantic = FakeRetriever([_result("semantic", 0.9)])
    lexical = FakeRetriever([_result("lexical", 0.8)])

    retriever = HybridRetriever(
        semantic,
        lexical,
        candidate_k=7,
    )

    await retriever.retrieve(
        "vehicle",
        top_k=2,
    )

    assert semantic.calls[0]["top_k"] == 7
    assert lexical.calls[0]["top_k"] == 7


@pytest.mark.asyncio
async def test_hybrid_retriever_propagates_filters_and_governance():
    semantic = FakeRetriever([_result("semantic", 0.9)])
    lexical = FakeRetriever([_result("lexical", 0.8)])

    policy = GovernancePolicy(
        required_metadata={"classification": "internal"},
    )

    retriever = HybridRetriever(semantic, lexical)

    await retriever.retrieve(
        "vehicle",
        metadata_filter={"source": "architecture.md"},
        governance_policy=policy,
    )

    for call in (semantic.calls[0], lexical.calls[0]):
        assert call["metadata_filter"] == {"source": "architecture.md"}
        assert call["governance_policy"] is policy


@pytest.mark.asyncio
async def test_hybrid_retriever_propagates_min_score():
    semantic = FakeRetriever([_result("semantic", 0.9)])
    lexical = FakeRetriever([_result("lexical", 0.8)])

    retriever = HybridRetriever(semantic, lexical)

    await retriever.retrieve(
        "vehicle",
        min_score=0.5,
    )

    assert semantic.calls[0]["min_score"] == 0.5
    assert lexical.calls[0]["min_score"] == 0.5


@pytest.mark.asyncio
async def test_hybrid_retriever_deduplicates_same_chunk():
    shared_semantic = _result("shared", 0.95)
    shared_lexical = _result("shared", 0.70)

    semantic = FakeRetriever([shared_semantic])
    lexical = FakeRetriever([shared_lexical])

    retriever = HybridRetriever(semantic, lexical)

    results = await retriever.retrieve(
        "vehicle",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.id == "shared"


@pytest.mark.asyncio
async def test_hybrid_retriever_is_deterministic_for_equal_fused_scores():
    semantic = FakeRetriever(
        [
            _result("b", 0.9),
            _result("a", 0.8),
        ]
    )
    lexical = FakeRetriever(
        [
            _result("a", 0.9),
            _result("b", 0.8),
        ]
    )

    retriever = HybridRetriever(
        semantic,
        lexical,
        candidate_k=2,
    )

    results = await retriever.retrieve(
        "vehicle",
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_hybrid_retriever_rejects_invalid_arguments():
    semantic = FakeRetriever([])
    lexical = FakeRetriever([])

    with pytest.raises(ValueError, match="candidate_k"):
        HybridRetriever(semantic, lexical, candidate_k=0)

    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(semantic, lexical, rrf_k=0)

    retriever = HybridRetriever(semantic, lexical)

    with pytest.raises(ValueError, match="empty"):
        await retriever.retrieve("")

    with pytest.raises(ValueError, match="top_k"):
        await retriever.retrieve("vehicle", top_k=0)

    with pytest.raises(ValueError, match="between -1.0 and 1.0"):
        await retriever.retrieve("vehicle", min_score=1.1)
