from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memory.retrieval.factory import MemoryRetrieverFactory
from memory.retrieval.lexical import LexicalMemoryRetriever
from memory.retrieval.hybrid import HybridMemoryRetriever
from memory.retrieval.postgres_lexical import PostgreSQLLexicalMemoryRetriever
from memory.retrieval.postgres_semantic import PostgreSQLSemanticMemoryRetriever
from memory.retrieval.reranking import RerankingMemoryRetriever


def test_factory_creates_lexical_retriever_for_in_memory() -> None:
    store = MagicMock()

    retriever = MemoryRetrieverFactory.create(
        backend="in_memory",
        memory_store=store,
    )

    assert isinstance(retriever, LexicalMemoryRetriever)
    assert retriever.store is store


def test_factory_creates_hybrid_retriever_for_postgres() -> None:
    embedding_service = MagicMock()
    store = MagicMock()

    retriever = MemoryRetrieverFactory.create(
        backend="postgres",
        memory_store=store,
        embedding_service=embedding_service,
    )

    assert isinstance(retriever, HybridMemoryRetriever)
    assert isinstance(
        retriever.semantic_retriever,
        PostgreSQLSemanticMemoryRetriever,
    )
    assert isinstance(
        retriever.lexical_retriever,
        PostgreSQLLexicalMemoryRetriever,
    )
    assert retriever.semantic_retriever._embedding_service is embedding_service


def test_factory_creates_reranking_retriever_for_cross_encoder() -> None:
    embedding_service = MagicMock()
    store = MagicMock()
    fake_reranker = MagicMock()

    with patch(
        "memory.retrieval.factory.CrossEncoderMemoryReranker",
        return_value=fake_reranker,
    ) as reranker_class:
        retriever = MemoryRetrieverFactory.create(
            backend="postgres",
            memory_store=store,
            embedding_service=embedding_service,
            reranker="cross_encoder",
            reranker_model_id="test-model",
            reranker_onnx_filename="test/model.onnx",
            reranker_max_length=4096,
            reranker_candidate_k=25,
        )

    assert isinstance(retriever, RerankingMemoryRetriever)
    assert isinstance(retriever.retriever, HybridMemoryRetriever)
    assert retriever.reranker is fake_reranker
    assert retriever.candidate_k == 25

    reranker_class.assert_called_once_with(
        model_id="test-model",
        onnx_filename="test/model.onnx",
        max_length=4096,
    )


def test_factory_does_not_load_reranker_when_disabled() -> None:
    with patch("memory.retrieval.factory.CrossEncoderMemoryReranker") as reranker_class:
        retriever = MemoryRetrieverFactory.create(
            backend="postgres",
            memory_store=MagicMock(),
            embedding_service=MagicMock(),
            reranker="none",
        )

    assert isinstance(retriever, HybridMemoryRetriever)
    reranker_class.assert_not_called()


def test_factory_rejects_unknown_reranker() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported memory reranker",
    ):
        MemoryRetrieverFactory.create(
            backend="postgres",
            memory_store=MagicMock(),
            embedding_service=MagicMock(),
            reranker="unsupported",
        )


def test_factory_rejects_invalid_reranker_max_length() -> None:
    with pytest.raises(
        ValueError,
        match="reranker_max_length must be greater than zero",
    ):
        MemoryRetrieverFactory.create(
            backend="postgres",
            memory_store=MagicMock(),
            embedding_service=MagicMock(),
            reranker_max_length=0,
        )


def test_factory_rejects_invalid_reranker_candidate_k() -> None:
    with pytest.raises(
        ValueError,
        match="reranker_candidate_k must be greater than zero",
    ):
        MemoryRetrieverFactory.create(
            backend="postgres",
            memory_store=MagicMock(),
            embedding_service=MagicMock(),
            reranker_candidate_k=0,
        )


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
