from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock

from app.config.settings import Settings
from app.control_plane.agent_runs.application_service import AgentRunApplicationService
from app.control_plane.agent_runs.recovery_service import AgentRunRecoveryService
from app.control_plane.dependencies import (
    _build_rag_retriever,
    get_agent_run_application_service,
    get_agent_run_recovery_service,
    get_mcp_server_lifecycle_service,
    get_usage_store,
)
from app.control_plane.usage.postgres_store import PostgreSQLUsageRepository
from app.control_plane.mcp_servers.lifecycle_service import MCPServerLifecycleService
from app.control_plane.mcp_servers.postgres_repository import (
    PostgreSQLMCPServerRepository,
)
from app.control_plane.tool_execution.postgres_idempotency import (
    PostgreSQLToolExecutionIdempotencyStore,
)
from rag.retrieval import HybridRetriever
from tools.execution.service import ToolExecutionService
from tools.mcp.config import MCPServerConfig


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


def test_get_mcp_server_lifecycle_service_uses_injected_session() -> None:
    from app.control_plane import dependencies

    session = Mock()

    service = get_mcp_server_lifecycle_service(db=session)

    assert isinstance(service, MCPServerLifecycleService)
    assert isinstance(
        service._repository,
        PostgreSQLMCPServerRepository,
    )
    assert service._repository._session is session
    assert service._manager is dependencies._mcp_server_manager


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
        agent_run_max_recovery_attempts=3,
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

    from app.control_plane import dependencies
    from app.control_plane.agent_run_steps.postgres_repository import (
        PostgreSQLAgentRunStepsRepository,
    )

    assert isinstance(
        service._agent_run_steps_repository,
        PostgreSQLAgentRunStepsRepository,
    )
    assert service._agent_run_steps_repository._session is not None
    assert service._cancellation_registry is dependencies._agent_run_cancellation_registry


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
        agent_run_max_recovery_attempts=3,
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

    from app.control_plane import dependencies

    assert service._cancellation_registry is dependencies._agent_run_cancellation_registry


@pytest.mark.asyncio
async def test_initialize_mcp_servers_registers_connects_and_discovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.control_plane import dependencies

    class FakeMCPManager:
        def __init__(self) -> None:
            self.registered: list[MCPServerConfig] = []
            self.connected: list[str] = []
            self.discovered: list[str] = []

        async def register_server(self, config: MCPServerConfig) -> None:
            self.registered.append(config)

        async def connect_and_discover(self, name: str) -> None:
            self.connected.append(name)
            self.discovered.append(name)

        async def disconnect_all(self) -> None:
            return None

    manager = FakeMCPManager()

    configured_settings = Settings(
        environment="test",
        aws_region="us-east-1",
        default_provider="mock",
        log_level="INFO",
        provider_credentials={},
        external_evaluation_release_required=False,
        agent_run_lease_duration_seconds=60,
        agent_run_max_recovery_attempts=3,
        mcp_servers=(
            MCPServerConfig(
                name="server-a",
                transport="stdio",
                command="python",
            ),
            MCPServerConfig(
                name="server-b",
                transport="streamable-http",
                url="http://127.0.0.1:9000/mcp",
            ),
        ),
    )

    monkeypatch.setattr(dependencies, "_mcp_server_manager", manager)
    monkeypatch.setattr(
        dependencies,
        "_mcp_servers_initialized",
        False,
    )
    monkeypatch.setattr(
        dependencies,
        "Settings",
        Mock(from_environment=Mock(return_value=configured_settings)),
    )

    await dependencies.initialize_mcp_servers()

    assert manager.registered == list(configured_settings.mcp_servers)
    assert manager.connected == ["server-a", "server-b"]
    assert manager.discovered == ["server-a", "server-b"]
    assert dependencies._mcp_servers_initialized is True


@pytest.mark.asyncio
async def test_initialize_mcp_servers_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.control_plane import dependencies

    manager = Mock()
    manager.register_server = AsyncMock()
    manager.connect_and_discover = AsyncMock()

    monkeypatch.setattr(dependencies, "_mcp_server_manager", manager)
    monkeypatch.setattr(
        dependencies,
        "_mcp_servers_initialized",
        True,
    )

    await dependencies.initialize_mcp_servers()

    manager.register_server.assert_not_awaited()
    manager.connect_and_discover.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_mcp_servers_disconnects_partial_startup_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.control_plane import dependencies

    manager = Mock()
    manager.register_server = AsyncMock()
    manager.connect_and_discover = AsyncMock(
        side_effect=[
            None,
            RuntimeError("server-b discovery failed"),
        ]
    )
    manager.disconnect_all = AsyncMock()

    configured_settings = Settings(
        environment="test",
        aws_region="us-east-1",
        default_provider="mock",
        log_level="INFO",
        provider_credentials={},
        external_evaluation_release_required=False,
        agent_run_lease_duration_seconds=60,
        agent_run_max_recovery_attempts=3,
        mcp_servers=(
            MCPServerConfig(
                name="server-a",
                transport="stdio",
                command="python",
            ),
            MCPServerConfig(
                name="server-b",
                transport="stdio",
                command="python",
            ),
        ),
    )

    monkeypatch.setattr(dependencies, "_mcp_server_manager", manager)
    monkeypatch.setattr(
        dependencies,
        "_mcp_servers_initialized",
        False,
    )
    monkeypatch.setattr(
        dependencies,
        "Settings",
        Mock(from_environment=Mock(return_value=configured_settings)),
    )

    with pytest.raises(
        RuntimeError,
        match="server-b discovery failed",
    ):
        await dependencies.initialize_mcp_servers()

    manager.disconnect_all.assert_awaited_once()
    assert dependencies._mcp_servers_initialized is False


@pytest.mark.asyncio
async def test_close_mcp_servers_disconnects_all_and_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.control_plane import dependencies

    manager = Mock()
    manager.disconnect_all = AsyncMock()

    monkeypatch.setattr(dependencies, "_mcp_server_manager", manager)
    monkeypatch.setattr(
        dependencies,
        "_mcp_servers_initialized",
        True,
    )

    await dependencies.close_mcp_servers()

    manager.disconnect_all.assert_awaited_once()
    assert dependencies._mcp_servers_initialized is False


def test_get_mcp_server_manager_returns_shared_manager() -> None:
    from app.control_plane import dependencies

    assert dependencies.get_mcp_server_manager() is dependencies._mcp_server_manager


@pytest.mark.asyncio
async def test_validate_agent_tool_capabilities_accepts_registered_enabled_tools(
    monkeypatch,
) -> None:
    from app.control_plane import dependencies

    class FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeAgent:
        def __init__(self) -> None:
            self.name = "test-agent"
            self.tool_names = ("test.tool",)

    class FakeToolRegistry:
        async def list_tools(self):
            return [FakeTool("test.tool")]

    class FakeAgentRegistry:
        async def list_agents(self):
            return [FakeAgent()]

    monkeypatch.setattr(
        dependencies,
        "_tool_registry",
        FakeToolRegistry(),
    )
    monkeypatch.setattr(
        dependencies,
        "_agent_registry",
        FakeAgentRegistry(),
    )

    await dependencies._validate_agent_tool_capabilities()


@pytest.mark.asyncio
async def test_validate_agent_tool_capabilities_rejects_missing_tool(
    monkeypatch,
) -> None:
    from app.control_plane import dependencies

    class FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeAgent:
        def __init__(self) -> None:
            self.name = "test-agent"
            self.tool_names = ("missing.tool",)

    class FakeToolRegistry:
        async def list_tools(self):
            return [FakeTool("existing.tool")]

    class FakeAgentRegistry:
        async def list_agents(self):
            return [FakeAgent()]

    monkeypatch.setattr(
        dependencies,
        "_tool_registry",
        FakeToolRegistry(),
    )
    monkeypatch.setattr(
        dependencies,
        "_agent_registry",
        FakeAgentRegistry(),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Agent 'test-agent' declares tool 'missing.tool', "
            "but that tool is not registered or enabled"
        ),
    ):
        await dependencies._validate_agent_tool_capabilities()


@pytest.mark.asyncio
async def test_validate_agent_tool_capabilities_rejects_disabled_tool(
    monkeypatch,
) -> None:
    from app.control_plane import dependencies

    class FakeTool:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeAgent:
        def __init__(self) -> None:
            self.name = "test-agent"
            self.tool_names = ("disabled.tool",)

    class FakeToolRegistry:
        async def list_tools(self):
            # ToolRegistry.list_tools() exposes enabled tools only.
            return []

    class FakeAgentRegistry:
        async def list_agents(self):
            return [FakeAgent()]

    monkeypatch.setattr(
        dependencies,
        "_tool_registry",
        FakeToolRegistry(),
    )
    monkeypatch.setattr(
        dependencies,
        "_agent_registry",
        FakeAgentRegistry(),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Agent 'test-agent' declares tool 'disabled.tool', "
            "but that tool is not registered or enabled"
        ),
    ):
        await dependencies._validate_agent_tool_capabilities()


def test_tool_execution_service_uses_configured_tenant_policy_engine() -> None:
    from app.control_plane import dependencies
    from ai_platform.agents.policy import TenantPolicyEngine

    assert isinstance(
        dependencies._tool_execution_service,
        ToolExecutionService,
    )

    if dependencies._tenant_policy_engine is not None:
        assert isinstance(
            dependencies._tenant_policy_engine,
            TenantPolicyEngine,
        )
        assert (
            dependencies._tool_execution_service.tenant_policy_engine
            is dependencies._tenant_policy_engine
        )


def test_build_tenant_policy_engine_loads_environment_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    from app.control_plane import dependencies
    from ai_platform.agents.policy import TenantPolicyEngine

    monkeypatch.setenv(
        "TENANT_POLICY_ENFORCEMENT_ENABLED",
        "true",
    )
    monkeypatch.setenv(
        "TENANT_POLICIES",
        json.dumps(
            [
                {
                    "tenant_id": "tenant-acme",
                    "allowed_tools": ["rag.search"],
                    "blocked_tools": ["vehicle.data.query"],
                    "allowed_mcp_servers": ["document-server"],
                }
            ]
        ),
    )

    engine = dependencies._build_tenant_policy_engine()

    assert isinstance(engine, TenantPolicyEngine)

    policy = engine.get_policy("tenant-acme")

    assert policy.tenant_id == "tenant-acme"
    assert policy.allowed_tools == frozenset({"rag.search"})
    assert policy.blocked_tools == frozenset({"vehicle.data.query"})
    assert policy.allowed_mcp_servers == frozenset({"document-server"})
