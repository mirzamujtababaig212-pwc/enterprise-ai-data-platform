from datetime import datetime, timedelta, timezone

import pytest

from memory.models import MemoryItem
from memory.retrieval.lexical import LexicalMemoryRetriever
from memory.stores.in_memory import InMemoryMemoryStore


def make_item(
    memory_id: str,
    content: str,
    *,
    memory_type: str = "semantic",
    namespace: str = "project-a",
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        memory_type=memory_type,
        content=content,
        namespace=namespace,
        created_at=created_at or datetime.now(timezone.utc),
        expires_at=expires_at,
    )


@pytest.mark.asyncio
async def test_retrieves_memories_by_lexical_relevance():
    store = InMemoryMemoryStore()

    await store.put(
        make_item(
            "memory-databricks",
            "The project uses Databricks for Spark workloads.",
        )
    )
    await store.put(
        make_item(
            "memory-snowflake",
            "The project uses Snowflake for the warehouse.",
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "Databricks Spark",
        namespace="project-a",
        top_k=5,
    )

    assert [item.id for item in results] == ["memory-databricks"]


@pytest.mark.asyncio
async def test_results_are_limited_by_top_k():
    store = InMemoryMemoryStore()

    await store.put(make_item("memory-1", "Databricks Spark platform"))
    await store.put(make_item("memory-2", "Databricks warehouse platform"))
    await store.put(make_item("memory-3", "Databricks data platform"))

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "Databricks platform",
        namespace="project-a",
        top_k=2,
    )

    assert len(results) == 2


@pytest.mark.asyncio
async def test_namespace_isolation_is_enforced():
    store = InMemoryMemoryStore()

    await store.put(
        make_item(
            "memory-project-a",
            "Databricks Spark platform",
            namespace="project-a",
        )
    )
    await store.put(
        make_item(
            "memory-project-b",
            "Databricks Spark platform",
            namespace="project-b",
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "Databricks Spark",
        namespace="project-a",
    )

    assert [item.id for item in results] == ["memory-project-a"]


@pytest.mark.asyncio
async def test_memory_type_filter_is_respected():
    store = InMemoryMemoryStore()

    await store.put(
        make_item(
            "memory-semantic",
            "Databricks architecture",
            memory_type="semantic",
        )
    )
    await store.put(
        make_item(
            "memory-episodic",
            "Discussed Databricks architecture",
            memory_type="episodic",
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "Databricks architecture",
        namespace="project-a",
        memory_type="semantic",
    )

    assert [item.id for item in results] == ["memory-semantic"]


@pytest.mark.asyncio
async def test_expired_memories_are_not_returned():
    store = InMemoryMemoryStore()

    await store.put(
        make_item(
            "memory-expired",
            "Databricks Spark platform",
            expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        )
    )
    await store.put(
        make_item(
            "memory-active",
            "Databricks Spark platform",
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "Databricks Spark",
        namespace="project-a",
    )

    assert [item.id for item in results] == ["memory-active"]


@pytest.mark.asyncio
async def test_results_are_deterministic_for_equal_relevance():
    store = InMemoryMemoryStore()

    timestamp = datetime.now(timezone.utc)

    await store.put(
        make_item(
            "memory-b",
            "Databricks platform",
            created_at=timestamp,
        )
    )
    await store.put(
        make_item(
            "memory-a",
            "Databricks platform",
            created_at=timestamp,
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "Databricks platform",
        namespace="project-a",
    )

    assert [item.id for item in results] == [
        "memory-a",
        "memory-b",
    ]


@pytest.mark.asyncio
async def test_no_matching_memory_returns_empty():
    store = InMemoryMemoryStore()

    await store.put(
        make_item(
            "memory-1",
            "The project uses Snowflake.",
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "Kubernetes",
        namespace="project-a",
    )

    assert results == ()


@pytest.mark.asyncio
async def test_empty_query_is_rejected():
    store = InMemoryMemoryStore()
    retriever = LexicalMemoryRetriever(store)

    with pytest.raises(
        ValueError,
        match="Query must not be empty",
    ):
        await retriever.retrieve(
            "",
            namespace="project-a",
        )


@pytest.mark.asyncio
async def test_empty_namespace_is_rejected():
    store = InMemoryMemoryStore()
    retriever = LexicalMemoryRetriever(store)

    with pytest.raises(
        ValueError,
        match="Memory namespace must not be empty",
    ):
        await retriever.retrieve(
            "Databricks",
            namespace="",
        )


@pytest.mark.asyncio
async def test_invalid_top_k_is_rejected():
    store = InMemoryMemoryStore()
    retriever = LexicalMemoryRetriever(store)

    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        await retriever.retrieve(
            "Databricks",
            namespace="project-a",
            top_k=0,
        )


@pytest.mark.asyncio
async def test_older_relevant_memory_is_retrieved_past_newer_irrelevant_memories():
    store = InMemoryMemoryStore()

    base_time = datetime.now(timezone.utc)

    await store.put(
        make_item(
            "memory-relevant",
            "Databricks Spark architecture decision",
            created_at=base_time - timedelta(days=10),
        )
    )

    for index in range(20):
        await store.put(
            make_item(
                f"memory-irrelevant-{index:02d}",
                "Unrelated project discussion",
                created_at=base_time - timedelta(days=index),
            )
        )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "Databricks Spark",
        namespace="project-a",
        top_k=1,
    )

    assert [item.id for item in results] == ["memory-relevant"]


@pytest.mark.asyncio
async def test_lexical_retriever_prefers_rare_terms_over_generic_terms():
    store = InMemoryMemoryStore()

    await store.put(
        make_item(
            "memory-generic",
            "production release deployment configuration monitoring",
        )
    )
    await store.put(
        make_item(
            "memory-specific",
            "deployment configuration requires validation before production release",
        )
    )
    await store.put(
        make_item(
            "memory-rare",
            "deployment configuration requires approval before production release",
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "deployment configuration approval",
        namespace="project-a",
        top_k=3,
    )

    assert results[0].id == "memory-rare"


@pytest.mark.asyncio
async def test_lexical_retriever_uses_term_frequency_for_equal_length_documents():
    store = InMemoryMemoryStore()
    base_time = datetime.now(timezone.utc)

    await store.put(
        make_item(
            "memory-repeated",
            "deployment deployment configuration",
            created_at=base_time,
        )
    )
    await store.put(
        make_item(
            "memory-single",
            "deployment release configuration",
            created_at=base_time + timedelta(minutes=1),
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "deployment",
        namespace="project-a",
        top_k=1,
    )

    assert [item.id for item in results] == ["memory-repeated"]


@pytest.mark.asyncio
async def test_lexical_retriever_uses_idf_for_rare_query_terms():
    store = InMemoryMemoryStore()
    base_time = datetime.now(timezone.utc)

    await store.put(
        make_item(
            "memory-generic",
            "deployment",
            created_at=base_time + timedelta(minutes=3),
        )
    )
    await store.put(
        make_item(
            "memory-generic-2",
            "deployment",
            created_at=base_time + timedelta(minutes=2),
        )
    )
    await store.put(
        make_item(
            "memory-generic-3",
            "deployment",
            created_at=base_time + timedelta(minutes=1),
        )
    )
    await store.put(
        make_item(
            "memory-rare",
            "approval",
            created_at=base_time,
        )
    )

    retriever = LexicalMemoryRetriever(store)

    results = await retriever.retrieve(
        "deployment approval",
        namespace="project-a",
        top_k=1,
    )

    assert [item.id for item in results] == ["memory-rare"]
