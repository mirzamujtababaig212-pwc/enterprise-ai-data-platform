import pytest

from common.config.settings import Settings
from rag.stores.factory import VectorStoreFactory
from rag.stores.in_memory import InMemoryVectorStore
from rag.stores.qdrant import QdrantVectorStore


def test_factory_defaults_to_in_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        Settings.vector_store,
        "BACKEND",
        "in_memory",
    )

    store = VectorStoreFactory.create()

    assert isinstance(store, InMemoryVectorStore)


def test_factory_creates_qdrant(monkeypatch) -> None:
    monkeypatch.setattr(
        Settings.vector_store,
        "BACKEND",
        "qdrant",
    )

    store = VectorStoreFactory.create()

    try:
        assert isinstance(store, QdrantVectorStore)
        assert store._collection_name == Settings.qdrant.COLLECTION
    finally:
        # The client has no network connection at construction time.
        # Close it explicitly to avoid leaving async resources open.
        import asyncio

        asyncio.run(store._client.close())


def test_factory_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        Settings.vector_store,
        "BACKEND",
        "unsupported",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported vector-store backend",
    ):
        VectorStoreFactory.create()
