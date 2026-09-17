from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memory.retrieval.factory import MemoryRetrieverFactory
from memory.retrieval.lexical import LexicalMemoryRetriever
from memory.retrieval.postgres_semantic import PostgreSQLSemanticMemoryRetriever


def test_factory_creates_lexical_retriever_for_in_memory() -> None:
    store = MagicMock()

    retriever = MemoryRetrieverFactory.create(
        backend="in_memory",
        memory_store=store,
    )

    assert isinstance(retriever, LexicalMemoryRetriever)
    assert retriever.store is store


def test_factory_creates_semantic_retriever_for_postgres() -> None:
    embedding_service = MagicMock()
    store = MagicMock()

    retriever = MemoryRetrieverFactory.create(
        backend="postgres",
        memory_store=store,
        embedding_service=embedding_service,
    )

    assert isinstance(retriever, PostgreSQLSemanticMemoryRetriever)
    assert retriever._embedding_service is embedding_service


def test_factory_requires_embedding_service_for_postgres() -> None:
    with pytest.raises(
        ValueError,
        match="embedding service is required",
    ):
        MemoryRetrieverFactory.create(
            backend="postgres",
            memory_store=MagicMock(),
        )


def test_factory_rejects_unknown_backend() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported memory-store backend",
    ):
        MemoryRetrieverFactory.create(
            backend="unsupported",
            memory_store=MagicMock(),
        )
