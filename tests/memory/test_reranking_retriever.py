from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory.models import MemoryItem
from memory.retrieval.hybrid import HybridMemoryRetriever
from memory.retrieval.reranking import RerankingMemoryRetriever


def _item(
    memory_id: str,
    content: str,
    *,
    memory_type: str = "semantic",
    namespace: str = "project-a",
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,  # type: ignore[arg-type]
        content=content,
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


class FakeMemoryReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def rerank(
        self,
        query: str,
        candidates: list[MemoryItem],
        *,
        top_k: int,
    ) -> list[MemoryItem]:
        self.calls.append(
            {
                "query": query,
                "candidates": candidates,
                "top_k": top_k,
            }
        )
        return list(reversed(candidates))[:top_k]


@pytest.mark.asyncio
async def test_reranking_memory_retriever_composes_retriever_and_reranker():
    retriever = FakeMemoryRetriever(
        [
            _item("a", "vehicle"),
            _item("b", "battery"),
            _item("c", "motor"),
        ]
    )
    reranker = FakeMemoryReranker()

    service = RerankingMemoryRetriever(
        retriever,
        reranker,
        candidate_k=3,
    )

    results = await service.retrieve(
        "vehicle battery",
        namespace="project-a",
        top_k=2,
    )

    assert [item.id for item in results] == ["c", "b"]

    assert retriever.calls[0]["top_k"] == 3
    assert reranker.calls[0]["top_k"] == 2
    assert reranker.calls[0]["query"] == "vehicle battery"


@pytest.mark.asyncio
async def test_reranking_memory_retriever_uses_top_k_when_candidate_k_not_set():
    retriever = FakeMemoryRetriever(
        [
            _item("a", "vehicle"),
            _item("b", "battery"),
        ]
    )
    reranker = FakeMemoryReranker()

    service = RerankingMemoryRetriever(retriever, reranker)

    await service.retrieve(
        "vehicle",
        namespace="project-a",
        top_k=2,
    )

    assert retriever.calls[0]["top_k"] == 2


@pytest.mark.asyncio
async def test_reranking_memory_retriever_propagates_namespace_and_type():
    retriever = FakeMemoryRetriever([_item("a", "vehicle")])
    reranker = FakeMemoryReranker()

    service = RerankingMemoryRetriever(retriever, reranker)

    await service.retrieve(
        "vehicle",
        namespace="project-a",
        memory_type="episodic",
    )

    call = retriever.calls[0]

    assert call["query"] == "vehicle"
    assert call["namespace"] == "project-a"
    assert call["memory_type"] == "episodic"


@pytest.mark.asyncio
async def test_reranking_memory_retriever_rejects_invalid_arguments():
    retriever = FakeMemoryRetriever([])
    reranker = FakeMemoryReranker()

    with pytest.raises(ValueError, match="candidate_k"):
        RerankingMemoryRetriever(
            retriever,
            reranker,
            candidate_k=0,
        )

    service = RerankingMemoryRetriever(retriever, reranker)

    with pytest.raises(ValueError, match="empty"):
        await service.retrieve(
            "",
            namespace="project-a",
        )

    with pytest.raises(ValueError, match="namespace"):
        await service.retrieve(
            "vehicle",
            namespace="",
        )

    with pytest.raises(ValueError, match="top_k"):
        await service.retrieve(
            "vehicle",
            namespace="project-a",
            top_k=0,
        )


@pytest.mark.asyncio
async def test_reranking_memory_retriever_composes_with_hybrid_retriever():
    semantic = FakeMemoryRetriever(
        [
            _item("semantic-first", "deployment configuration"),
            _item("shared", "production deployment"),
            _item("semantic-third", "deployment checklist"),
        ]
    )
    lexical = FakeMemoryRetriever(
        [
            _item("shared", "production deployment"),
            _item("lexical-second", "deployment release"),
            _item("lexical-third", "release monitoring"),
        ]
    )

    hybrid = HybridMemoryRetriever(
        semantic,
        lexical,
        candidate_k=3,
    )

    reranker = FakeMemoryReranker()
    service = RerankingMemoryRetriever(
        hybrid,
        reranker,
        candidate_k=3,
    )

    results = await service.retrieve(
        "deployment release",
        namespace="project-a",
        top_k=2,
    )

    assert len(results) == 2
    assert reranker.calls[0]["query"] == "deployment release"
    assert reranker.calls[0]["top_k"] == 2

    candidates = reranker.calls[0]["candidates"]
    assert [item.id for item in candidates] == [
        "shared",
        "semantic-first",
        "semantic-third",
    ]
