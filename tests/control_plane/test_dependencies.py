from __future__ import annotations

import pytest
from unittest.mock import Mock

from app.control_plane.dependencies import (
    _build_rag_retriever,
    get_usage_store,
)
from app.control_plane.usage.postgres_store import PostgreSQLUsageRepository
from rag.retrieval import HybridRetriever


def test_get_usage_store_creates_repository_from_injected_session() -> None:
    session_one = Mock()
    session_two = Mock()

    store_one = get_usage_store(db=session_one)
    store_two = get_usage_store(db=session_two)

    assert isinstance(store_one, PostgreSQLUsageRepository)
    assert isinstance(store_two, PostgreSQLUsageRepository)

    assert store_one is not store_two
    assert store_one._session is session_one
    assert store_two._session is session_two


def test_get_external_evaluation_dispatcher_is_lazy() -> None:
    from app.control_plane.dependencies import get_external_evaluation_dispatcher

    dispatcher = get_external_evaluation_dispatcher()

    assert dispatcher is not None
    assert dispatcher._faithfulness_workflow is None
    assert dispatcher._faithfulness_workflow_factory is not None


def test_build_rag_retriever_keeps_semantic_retriever_for_non_hybrid_backends() -> None:
    semantic_retriever = Mock()
    lexical_retriever = Mock()

    for backend in ("in_memory",):
        result = _build_rag_retriever(
            semantic_retriever=semantic_retriever,
            backend=backend,
            lexical_retriever=lexical_retriever,
        )

        assert result is semantic_retriever


def test_build_rag_retriever_uses_hybrid_retriever_for_postgres() -> None:
    semantic_retriever = Mock()
    lexical_retriever = Mock()

    result = _build_rag_retriever(
        semantic_retriever=semantic_retriever,
        backend="postgres",
        lexical_retriever=lexical_retriever,
    )

    assert isinstance(result, HybridRetriever)
    assert result.semantic_retriever is semantic_retriever
    assert result.lexical_retriever is lexical_retriever
    assert result.candidate_k == 5
    assert result.rrf_k == 60
    assert result.semantic_weight == 1.0
    assert result.lexical_weight == 0.5


def test_build_rag_retriever_uses_hybrid_retriever_for_qdrant() -> None:
    semantic_retriever = Mock()
    lexical_retriever = Mock()

    result = _build_rag_retriever(
        semantic_retriever=semantic_retriever,
        backend="qdrant",
        lexical_retriever=lexical_retriever,
    )

    assert isinstance(result, HybridRetriever)
    assert result.semantic_retriever is semantic_retriever
    assert result.lexical_retriever is lexical_retriever


def test_build_rag_retriever_requires_lexical_retriever_for_hybrid_backend() -> None:
    with pytest.raises(
        ValueError,
        match="PostgreSQL lexical retriever is required",
    ):
        _build_rag_retriever(
            semantic_retriever=Mock(),
            backend="postgres",
        )
