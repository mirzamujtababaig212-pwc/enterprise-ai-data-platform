import pytest

from rag.governance import GovernancePolicy
from rag.models import DocumentChunk
from rag.retrieval import InMemoryLexicalRetriever


def _chunks() -> tuple[DocumentChunk, ...]:
    return (
        DocumentChunk(
            id="electric",
            document_id="vehicle",
            content="Electric vehicles use battery powered electric motors.",
            metadata={"classification": "internal"},
        ),
        DocumentChunk(
            id="gasoline",
            document_id="vehicle",
            content="Gasoline vehicles use internal combustion engines.",
            metadata={"classification": "internal"},
        ),
        DocumentChunk(
            id="restricted-electric",
            document_id="vehicle",
            content="Electric vehicle battery architecture.",
            metadata={"classification": "restricted"},
        ),
    )


@pytest.mark.asyncio
async def test_lexical_retriever_returns_exact_term_match_first():
    retriever = InMemoryLexicalRetriever(_chunks())

    results = await retriever.retrieve(
        "electric vehicle battery",
        top_k=2,
    )

    assert [result.chunk.id for result in results] == [
        "restricted-electric",
        "electric",
    ]
    assert all(0.0 <= result.score <= 1.0 for result in results)


@pytest.mark.asyncio
async def test_lexical_retriever_rejects_empty_query():
    retriever = InMemoryLexicalRetriever(_chunks())

    with pytest.raises(ValueError, match="empty"):
        await retriever.retrieve("")


@pytest.mark.asyncio
async def test_lexical_retriever_applies_metadata_filter():
    retriever = InMemoryLexicalRetriever(_chunks())

    results = await retriever.retrieve(
        "electric vehicle",
        top_k=5,
        metadata_filter={"classification": "internal"},
    )

    assert [result.chunk.id for result in results] == ["electric"]


@pytest.mark.asyncio
async def test_lexical_retriever_applies_governance_policy():
    retriever = InMemoryLexicalRetriever(_chunks())

    policy = GovernancePolicy(
        required_metadata={"classification": "internal"},
    )

    results = await retriever.retrieve(
        "electric vehicle",
        top_k=5,
        governance_policy=policy,
    )

    assert "restricted-electric" not in {result.chunk.id for result in results}


@pytest.mark.asyncio
async def test_lexical_retriever_rejects_conflicting_governance_policy():
    retriever = InMemoryLexicalRetriever(_chunks())

    policy = GovernancePolicy(
        required_metadata={"classification": "internal"},
    )

    with pytest.raises(ValueError, match="conflicts with governance policy"):
        await retriever.retrieve(
            "electric vehicle",
            metadata_filter={"classification": "restricted"},
            governance_policy=policy,
        )


@pytest.mark.asyncio
async def test_lexical_retriever_is_deterministic_for_equal_scores():
    chunks = (
        DocumentChunk(
            id="b",
            document_id="doc",
            content="electric vehicle",
        ),
        DocumentChunk(
            id="a",
            document_id="doc",
            content="electric vehicle",
        ),
    )

    retriever = InMemoryLexicalRetriever(chunks)

    results = await retriever.retrieve(
        "electric vehicle",
        top_k=2,
    )

    assert [result.chunk.id for result in results] == ["a", "b"]


@pytest.mark.asyncio
async def test_lexical_retriever_returns_empty_for_no_match():
    retriever = InMemoryLexicalRetriever(_chunks())

    results = await retriever.retrieve(
        "quantum computing",
        top_k=5,
    )

    assert results == []
