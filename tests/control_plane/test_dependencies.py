from __future__ import annotations

import pytest
from unittest.mock import Mock

from app.config.settings import Settings
from app.control_plane.agent_runs.application_service import AgentRunApplicationService
from app.control_plane.agent_runs.recovery_service import AgentRunRecoveryService
from app.control_plane.dependencies import (
    _build_rag_retriever,
    get_agent_run_application_service,
    get_agent_run_recovery_service,
    get_usage_store,
)
from app.control_plane.usage.postgres_store import PostgreSQLUsageRepository
from app.control_plane.tool_execution.postgres_idempotency import (
    PostgreSQLToolExecutionIdempotencyStore,
)
from rag.retrieval import HybridRetriever
from tools.execution.service import ToolExecutionService


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


def test_tool_execution_service_uses_durable_idempotency_store() -> None:
    from app.control_plane import dependencies

    assert isinstance(dependencies._tool_execution_service, ToolExecutionService)
    assert isinstance(
        dependencies._tool_execution_service.idempotency_store,
        PostgreSQLToolExecutionIdempotencyStore,
    )
    assert (
        dependencies._tool_execution_service.idempotency_store._session_factory
        is dependencies.SessionLocal
    )


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


@pytest.mark.asyncio
async def test_get_agent_run_application_service_uses_configured_lease_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_settings = Settings(
        environment="test",
        aws_region="us-east-1",
        default_provider="mock",
        log_level="INFO",
        provider_credentials={},
        external_evaluation_release_required=False,
        agent_run_lease_duration_seconds=123,
    )

    monkeypatch.setattr(
        "app.control_plane.dependencies.Settings.from_environment",
        classmethod(lambda cls: configured_settings),
    )

    async def initialize_agents() -> None:
        return None

    monkeypatch.setattr(
        "app.control_plane.dependencies._initialize_agents",
        initialize_agents,
    )

    service = await get_agent_run_application_service(db=Mock())

    assert isinstance(service, AgentRunApplicationService)
    assert service._lease_seconds == 123


@pytest.mark.asyncio
async def test_get_agent_run_recovery_service_uses_configured_lease_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_settings = Settings(
        environment="test",
        aws_region="us-east-1",
        default_provider="mock",
        log_level="INFO",
        provider_credentials={},
        external_evaluation_release_required=False,
        agent_run_lease_duration_seconds=123,
    )

    monkeypatch.setattr(
        "app.control_plane.dependencies.Settings.from_environment",
        classmethod(lambda cls: configured_settings),
    )

    async def initialize_agents() -> None:
        return None

    monkeypatch.setattr(
        "app.control_plane.dependencies._initialize_agents",
        initialize_agents,
    )

    service = await get_agent_run_recovery_service(db=Mock())

    assert isinstance(service, AgentRunRecoveryService)
    assert service._lease_seconds == 123
