from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory.models import MemoryItem
from memory.retrieval.hybrid import HybridMemoryRetriever


def _item(
    memory_id: str,
    *,
    memory_type: str = "semantic",
    namespace: str = "project-a",
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,  # type: ignore[arg-type]
        content=f"content for {memory_id}",
        namespace=namespace,
        created_at=datetime.now(timezone.utc),
    )


class FakeMemoryRetriever:
    def __init__(self, results: list[MemoryItem]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        query: str,
        *,
        namespace: str,
        memory_type: str | None = None,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        self.calls.append(
            {
                "query": query,
                "namespace": namespace,
                "memory_type": memory_type,
                "top_k": top_k,
            }
        )
        return self.results[:top_k]


@pytest.mark.asyncio
async def test_hybrid_memory_retriever_fuses_overlapping_results_with_rrf():
    semantic = FakeMemoryRetriever(
        [
            _item("shared"),
            _item("semantic-only"),
            _item("semantic-third"),
        ]
    )
    lexical = FakeMemoryRetriever(
        [
            _item("lexical-only"),
            _item("shared"),
            _item("lexical-third"),
        ]
    )

    retriever = HybridMemoryRetriever(
        semantic,
        lexical,
        candidate_k=3,
        rrf_k=60,
    )

    results = await retriever.retrieve(
        "deployment configuration",
        namespace="project-a",
        top_k=4,
    )

    assert [item.id for item in results] == [
        "shared",
        "semantic-only",
        "semantic-third",
        "lexical-only",
    ]


@pytest.mark.asyncio
async def test_hybrid_memory_retriever_applies_custom_weights():
    semantic = FakeMemoryRetriever([_item("semantic")])
    lexical = FakeMemoryRetriever([_item("lexical")])

    retriever = HybridMemoryRetriever(
        semantic,
        lexical,
        semantic_weight=2.0,
        lexical_weight=0.25,
        rrf_k=60,
    )

    results = await retriever.retrieve(
        "deployment",
        namespace="project-a",
        top_k=2,
    )

    assert [item.id for item in results] == [
        "semantic",
        "lexical",
    ]


@pytest.mark.asyncio
async def test_hybrid_memory_retriever_uses_candidate_k_for_both_retrievers():
    semantic = FakeMemoryRetriever([_item("semantic")])
    lexical = FakeMemoryRetriever([_item("lexical")])

    retriever = HybridMemoryRetriever(
        semantic,
        lexical,
        candidate_k=7,
    )

    await retriever.retrieve(
        "deployment",
        namespace="project-a",
        top_k=2,
    )

    assert semantic.calls[0]["top_k"] == 7
    assert lexical.calls[0]["top_k"] == 7


@pytest.mark.asyncio
async def test_hybrid_memory_retriever_propagates_namespace_and_type():
    semantic = FakeMemoryRetriever([_item("semantic")])
    lexical = FakeMemoryRetriever([_item("lexical")])

    retriever = HybridMemoryRetriever(semantic, lexical)

    await retriever.retrieve(
        "deployment",
        namespace="project-a",
        memory_type="episodic",
    )

    for call in (semantic.calls[0], lexical.calls[0]):
        assert call["query"] == "deployment"
        assert call["namespace"] == "project-a"
        assert call["memory_type"] == "episodic"


@pytest.mark.asyncio
async def test_hybrid_memory_retriever_deduplicates_same_memory():
    semantic = FakeMemoryRetriever([_item("shared")])
    lexical = FakeMemoryRetriever([_item("shared")])

    retriever = HybridMemoryRetriever(semantic, lexical)

    results = await retriever.retrieve(
        "deployment",
        namespace="project-a",
    )

    assert len(results) == 1
    assert results[0].id == "shared"


@pytest.mark.asyncio
async def test_hybrid_memory_retriever_is_deterministic_for_equal_fused_scores():
    semantic = FakeMemoryRetriever(
        [
            _item("b"),
            _item("a"),
        ]
    )
    lexical = FakeMemoryRetriever(
        [
            _item("a"),
            _item("b"),
        ]
    )

    retriever = HybridMemoryRetriever(
        semantic,
        lexical,
        candidate_k=2,
        semantic_weight=1.0,
        lexical_weight=1.0,
    )

    results = await retriever.retrieve(
        "deployment",
        namespace="project-a",
        top_k=2,
    )

    assert [item.id for item in results] == [
        "a",
        "b",
    ]


@pytest.mark.asyncio
async def test_hybrid_memory_retriever_rejects_invalid_arguments():
    semantic = FakeMemoryRetriever([])
    lexical = FakeMemoryRetriever([])

    with pytest.raises(ValueError, match="candidate_k"):
        HybridMemoryRetriever(semantic, lexical, candidate_k=0)

    with pytest.raises(ValueError, match="rrf_k"):
        HybridMemoryRetriever(semantic, lexical, rrf_k=0)

    with pytest.raises(ValueError, match="semantic_weight"):
        HybridMemoryRetriever(semantic, lexical, semantic_weight=-0.1)

    with pytest.raises(ValueError, match="lexical_weight"):
        HybridMemoryRetriever(semantic, lexical, lexical_weight=-0.1)

    with pytest.raises(ValueError, match="At least one retrieval weight"):
        HybridMemoryRetriever(
            semantic,
            lexical,
            semantic_weight=0.0,
            lexical_weight=0.0,
        )

    retriever = HybridMemoryRetriever(semantic, lexical)

    with pytest.raises(ValueError, match="empty"):
        await retriever.retrieve(
            "",
            namespace="project-a",
        )

    with pytest.raises(ValueError, match="namespace"):
        await retriever.retrieve(
            "deployment",
            namespace="",
        )

    with pytest.raises(ValueError, match="top_k"):
        await retriever.retrieve(
            "deployment",
            namespace="project-a",
            top_k=0,
        )
