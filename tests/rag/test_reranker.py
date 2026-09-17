import pytest

from rag.models import DocumentChunk, RetrievalResult
from rag.retrieval import TokenOverlapReranker


def _result(
    chunk_id: str,
    content: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk=DocumentChunk(
            id=chunk_id,
            document_id=f"doc-{chunk_id}",
            content=content,
        ),
        score=score,
    )


@pytest.mark.asyncio
async def test_token_overlap_reranker_reorders_candidates():
    reranker = TokenOverlapReranker()

    results = await reranker.rerank(
        "electric battery",
        [
            _result("weak", "vehicle maintenance", 0.9),
            _result("battery", "battery storage system", 0.8),
            _result("electric", "electric motor", 0.7),
        ],
        top_k=3,
    )

    assert [result.chunk.id for result in results] == [
        "battery",
        "electric",
        "weak",
    ]


@pytest.mark.asyncio
async def test_token_overlap_reranker_returns_requested_top_k():
    reranker = TokenOverlapReranker()

    results = await reranker.rerank(
        "vehicle",
        [
            _result("a", "vehicle battery", 0.9),
            _result("b", "vehicle motor", 0.8),
            _result("c", "vehicle brakes", 0.7),
        ],
        top_k=2,
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_token_overlap_reranker_preserves_embedding_identity():
    from rag.models import EmbeddingIdentity

    identity = EmbeddingIdentity(
        requested_provider="test",
        requested_model="test-model",
        resolved_provider="test",
        resolved_model="test-model",
        dimension=2,
    )

    result = RetrievalResult(
        chunk=DocumentChunk(
            id="chunk",
            document_id="doc",
            content="electric battery",
        ),
        score=0.8,
        embedding_identity=identity,
    )

    reranked = await TokenOverlapReranker().rerank(
        "electric",
        [result],
        top_k=1,
    )

    assert reranked[0].embedding_identity is identity


@pytest.mark.asyncio
async def test_token_overlap_reranker_is_deterministic_for_equal_scores():
    reranker = TokenOverlapReranker()

    results = await reranker.rerank(
        "vehicle",
        [
            _result("first", "vehicle", 0.9),
            _result("second", "vehicle", 0.8),
        ],
        top_k=2,
    )

    assert [result.chunk.id for result in results] == [
        "first",
        "second",
    ]


@pytest.mark.asyncio
async def test_token_overlap_reranker_rejects_invalid_arguments():
    reranker = TokenOverlapReranker()
    candidate = _result("chunk", "vehicle", 0.8)

    with pytest.raises(ValueError, match="empty"):
        await reranker.rerank("", [candidate], top_k=1)

    with pytest.raises(ValueError, match="top_k"):
        await reranker.rerank("vehicle", [candidate], top_k=0)
