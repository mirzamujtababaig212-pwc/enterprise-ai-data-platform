from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from dataclasses import replace

import pytest

from tools.mcp.config import MCPServerConfig, MCPToolCapability
from tools.mcp.health import MCPHealthHistory
from tools.mcp.manager import MCPServerManager
from tools.mcp.models import MCPToolDefinition
from tools.mcp.recovery import MCPRecoveryPolicy
from tools.models import ToolDefinition
from tools.registry.in_memory import InMemoryToolRegistry


def make_stdio_config(
    name: str = "test-server",
    **kwargs,
) -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport="stdio",
        command="python",
        args=("server.py",),
        **kwargs,
    )


def make_manager() -> MCPServerManager:
    return MCPServerManager(InMemoryToolRegistry())


def make_mcp_tool(
    name: str = "search",
    description: str = "Search documents",
    input_schema: dict | None = None,
) -> MCPToolDefinition:
    return MCPToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema if input_schema is not None else {"type": "object"},
    )


def test_manager_starts_with_no_servers():
    manager = make_manager()

    assert manager.list_servers() == []


@pytest.mark.asyncio
async def test_register_server():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    assert manager.list_servers() == ["server-a"]
    assert manager.is_connected("server-a") is False


@pytest.mark.asyncio
async def test_register_server_rejects_duplicate_name():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        await manager.register_server(make_stdio_config("server-a"))


@pytest.mark.asyncio
async def test_unknown_server_raises_key_error():
    manager = make_manager()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        manager.get_config("missing-server")


@pytest.mark.asyncio
async def test_unknown_server_connection_raises_key_error():
    manager = make_manager()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        await manager.connect_server("missing-server")


@pytest.mark.asyncio
async def test_empty_server_name_raises_value_error():
    manager = make_manager()

    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        manager.is_connected(" ")


@pytest.mark.asyncio
async def test_register_server_rejects_unsupported_transport():
    manager = make_manager()

    config = MCPServerConfig(
        name="http-server",
        transport="unsupported",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported MCP server transport",
    ):
        await manager.register_server(config)


@pytest.mark.asyncio
async def test_connect_server_updates_connection_state():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()

    await manager.connect_server("server-a")

    client.connect.assert_awaited_once()
    assert manager.is_connected("server-a") is True


@pytest.mark.asyncio
async def test_connect_server_is_idempotent():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()

    await manager.connect_server("server-a")
    await manager.connect_server("server-a")

    client.connect.assert_awaited_once()
    assert manager.is_connected("server-a") is True


@pytest.mark.asyncio
async def test_connect_failure_does_not_leave_server_connected():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")

    client.connect = AsyncMock(side_effect=RuntimeError("connection failed"))

    with pytest.raises(
        RuntimeError,
        match="connection failed",
    ):
        await manager.connect_server("server-a")

    assert manager.is_connected("server-a") is False


@pytest.mark.asyncio
async def test_discover_requires_connection():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    with pytest.raises(
        RuntimeError,
        match="is not connected",
    ):
        await manager.discover_server("server-a")


@pytest.mark.asyncio
async def test_discover_server_registers_tools():
    registry = InMemoryToolRegistry()
    manager = MCPServerManager(registry)

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()
    client.list_tools = AsyncMock(
        return_value=[
            make_mcp_tool(
                name="search",
                description="Search documents",
                input_schema={"type": "object"},
            )
        ]
    )

    await manager.connect_server("server-a")

    definitions = await manager.discover_server("server-a")

    assert definitions == [
        ToolDefinition(
            name="search",
            description="Search documents",
            input_schema={"type": "object"},
            metadata={
                "source": "mcp",
                "mcp_server": "server-a",
                "capability": "unclassified",
                "risk_tier": "unknown",
                "side_effect": True,
            },
        )
    ]

    registered = await registry.get("search")

    assert registered is not None
    assert registered.definition.name == "search"
    assert registered.definition.metadata == {
        "source": "mcp",
        "mcp_server": "server-a",
        "capability": "unclassified",
        "risk_tier": "unknown",
        "side_effect": True,
    }


@pytest.mark.asyncio
async def test_connect_and_discover():
    registry = InMemoryToolRegistry()
    manager = MCPServerManager(registry)

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()
    client.list_tools = AsyncMock(
        return_value=[
            make_mcp_tool(
                name="search",
                description="Search documents",
                input_schema={"type": "object"},
            )
        ]
    )

    definitions = await manager.connect_and_discover("server-a")

    assert manager.is_connected("server-a") is True
    assert len(definitions) == 1
    assert definitions[0].name == "search"


@pytest.mark.asyncio
async def test_discover_server_propagates_configured_tool_capability():
    registry = InMemoryToolRegistry()
    manager = MCPServerManager(registry)

    config = MCPServerConfig(
        name="server-a",
        transport="stdio",
        command="python",
        args=("server.py",),
        tool_capabilities={
            "search": MCPToolCapability(
                capability="document.read",
                risk_tier="low",
                side_effect=False,
                permission_scope="document:read",
            ),
        },
    )

    await manager.register_server(config)

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()
    client.list_tools = AsyncMock(
        return_value=[
            make_mcp_tool(
                name="search",
                description="Search documents",
                input_schema={"type": "object"},
            )
        ]
    )

    await manager.connect_server("server-a")

    definitions = await manager.discover_server("server-a")

    assert definitions[0].metadata == {
        "source": "mcp",
        "mcp_server": "server-a",
        "capability": "document.read",
        "risk_tier": "low",
        "side_effect": False,
        "permission_scope": "document:read",
    }


@pytest.mark.asyncio
async def test_unregister_server_removes_disconnected_server_and_owned_tools():
    registry = InMemoryToolRegistry()
    manager = MCPServerManager(registry)

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")
    client.list_tools = AsyncMock(
        return_value=[
            make_mcp_tool(name="search"),
            make_mcp_tool(name="get_document"),
        ]
    )

    client.connect = AsyncMock()
    await manager.connect_server("server-a")
    await manager.discover_server("server-a")

    await manager.unregister_server("server-a")

    assert manager.list_servers() == []
    assert await registry.get("search") is None
    assert await registry.get("get_document") is None


@pytest.mark.asyncio
async def test_unregister_server_removes_disabled_owned_tools():
    registry = InMemoryToolRegistry()
    manager = MCPServerManager(registry)

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()
    client.list_tools = AsyncMock(return_value=[make_mcp_tool(name="search")])

    await manager.connect_and_discover("server-a")

    tool = await registry.get("search")
    assert tool is not None

    disabled_definition = replace(
        tool.definition,
        enabled=False,
    )

    class DisabledTool:
        definition = disabled_definition

        async def execute(self, arguments):
            return await tool.execute(arguments)

    await registry.register(DisabledTool())

    assert await registry.list_tools() == []

    await manager.unregister_server("server-a")

    assert await registry.get("search") is None
    assert manager.list_servers() == []


@pytest.mark.asyncio
async def test_unregister_server_disconnects_connected_server_first():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()

    await manager.connect_server("server-a")
    await manager.unregister_server("server-a")

    client.disconnect.assert_awaited_once()
    assert manager.list_servers() == []


@pytest.mark.asyncio
async def test_unregister_server_preserves_other_server_and_tools():
    registry = InMemoryToolRegistry()
    manager = MCPServerManager(registry)

    await manager.register_server(make_stdio_config("server-a"))
    await manager.register_server(make_stdio_config("server-b"))

    client_a = await manager.get_client("server-a")
    client_b = await manager.get_client("server-b")

    client_a.connect = AsyncMock()
    client_b.connect = AsyncMock()

    client_a.list_tools = AsyncMock(return_value=[make_mcp_tool(name="search")])
    client_b.list_tools = AsyncMock(return_value=[make_mcp_tool(name="policy")])

    await manager.connect_and_discover("server-a")
    await manager.connect_and_discover("server-b")

    await manager.unregister_server("server-a")

    assert manager.list_servers() == ["server-b"]
    assert await registry.get("search") is None
    assert await registry.get("policy") is not None
    assert manager.is_connected("server-b") is True


@pytest.mark.asyncio
async def test_unregister_server_rejects_unknown_server():
    manager = make_manager()

    with pytest.raises(
        KeyError,
        match="not registered",
    ):
        await manager.unregister_server("missing-server")


@pytest.mark.asyncio
async def test_unregister_server_preserves_registration_when_disconnect_fails():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()
    client.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))

    await manager.connect_server("server-a")

    with pytest.raises(
        RuntimeError,
        match="disconnect failed",
    ):
        await manager.unregister_server("server-a")

    assert manager.list_servers() == ["server-a"]
    assert manager.is_connected("server-a") is False


@pytest.mark.asyncio
async def test_disconnect_server():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()
    client.disconnect = AsyncMock()

    await manager.connect_server("server-a")
    await manager.disconnect_server("server-a")

    client.disconnect.assert_awaited_once()
    assert manager.is_connected("server-a") is False


@pytest.mark.asyncio
async def test_disconnect_server_is_idempotent():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()
    client.disconnect = AsyncMock()

    await manager.connect_server("server-a")
    await manager.disconnect_server("server-a")
    await manager.disconnect_server("server-a")

    client.disconnect.assert_awaited_once()
    assert manager.is_connected("server-a") is False


@pytest.mark.asyncio
async def test_disconnect_failure_still_marks_server_disconnected():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()
    client.disconnect = AsyncMock(side_effect=RuntimeError("disconnect failed"))

    await manager.connect_server("server-a")

    with pytest.raises(
        RuntimeError,
        match="disconnect failed",
    ):
        await manager.disconnect_server("server-a")

    assert manager.is_connected("server-a") is False


@pytest.mark.asyncio
async def test_disconnect_all_disconnects_every_server():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))
    await manager.register_server(make_stdio_config("server-b"))

    client_a = await manager.get_client("server-a")
    client_b = await manager.get_client("server-b")

    client_a.connect = AsyncMock()
    client_b.connect = AsyncMock()

    client_a.disconnect = AsyncMock()
    client_b.disconnect = AsyncMock()

    await manager.connect_server("server-a")
    await manager.connect_server("server-b")

    await manager.disconnect_all()

    client_a.disconnect.assert_awaited_once()
    client_b.disconnect.assert_awaited_once()

    assert manager.is_connected("server-a") is False
    assert manager.is_connected("server-b") is False


@pytest.mark.asyncio
async def test_disconnect_all_attempts_remaining_servers_after_failure():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))
    await manager.register_server(make_stdio_config("server-b"))

    client_a = await manager.get_client("server-a")
    client_b = await manager.get_client("server-b")

    client_a.connect = AsyncMock()
    client_b.connect = AsyncMock()

    client_a.disconnect = AsyncMock(side_effect=RuntimeError("server-a disconnect failed"))
    client_b.disconnect = AsyncMock()

    await manager.connect_server("server-a")
    await manager.connect_server("server-b")

    with pytest.raises(
        RuntimeError,
        match="server-a disconnect failed",
    ):
        await manager.disconnect_all()

    client_a.disconnect.assert_awaited_once()
    client_b.disconnect.assert_awaited_once()

    assert manager.is_connected("server-a") is False
    assert manager.is_connected("server-b") is False


@pytest.mark.asyncio
async def test_disconnect_all_uses_reverse_registration_order():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))
    await manager.register_server(make_stdio_config("server-b"))

    client_a = await manager.get_client("server-a")
    client_b = await manager.get_client("server-b")

    client_a.connect = AsyncMock()
    client_b.connect = AsyncMock()

    disconnect_order: list[str] = []

    async def disconnect_a() -> None:
        disconnect_order.append("server-a")

    async def disconnect_b() -> None:
        disconnect_order.append("server-b")

    client_a.disconnect = AsyncMock(side_effect=disconnect_a)
    client_b.disconnect = AsyncMock(side_effect=disconnect_b)

    await manager.connect_server("server-a")
    await manager.connect_server("server-b")

    await manager.disconnect_all()

    assert disconnect_order == [
        "server-b",
        "server-a",
    ]


def test_health_history_transitions_ordinary_failures_and_recovery():
    history = MCPHealthHistory()

    checked_at = datetime.now(timezone.utc)

    assert history.record_success(checked_at).value == "healthy"

    assert (
        history.record_failure(
            checked_at,
            error="temporary failure",
        ).value
        == "degraded"
    )

    assert (
        history.record_failure(
            checked_at,
            error="temporary failure",
        ).value
        == "unhealthy"
    )

    assert history.record_success(checked_at).value == "degraded"
    assert history.record_success(checked_at).value == "healthy"


def test_health_history_hard_failure_is_immediately_unhealthy():
    history = MCPHealthHistory()

    checked_at = datetime.now(timezone.utc)

    assert history.record_success(checked_at).value == "healthy"

    assert (
        history.record_failure(
            checked_at,
            error="server disconnected",
            hard_failure=True,
        ).value
        == "unhealthy"
    )


def test_health_history_resets_opposite_consecutive_counter():
    history = MCPHealthHistory()

    checked_at = datetime.now(timezone.utc)

    history.record_failure(
        checked_at,
        error="temporary failure",
    )

    assert history.consecutive_failures == 1
    assert history.consecutive_successes == 0

    history.record_success(checked_at)

    assert history.consecutive_failures == 0
    assert history.consecutive_successes == 1
    assert history.last_error is None


@pytest.mark.asyncio
async def test_check_health_reports_healthy_connected_server():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()
    client.send_ping = AsyncMock()

    await manager.connect_server("server-a")

    result = await manager.check_health("server-a")

    client.send_ping.assert_awaited_once()
    assert result.server_name == "server-a"
    assert result.status.value == "healthy"
    assert result.latency_ms >= 0.0
    assert result.last_check.tzinfo is not None
    assert result.error is None


@pytest.mark.asyncio
async def test_check_health_reports_unhealthy_when_server_is_disconnected():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    result = await manager.check_health("server-a")

    assert result.server_name == "server-a"
    assert result.status.value == "unhealthy"
    assert result.latency_ms == 0.0
    assert result.error == "MCP server is not connected."
    assert result.last_check.tzinfo is not None


@pytest.mark.asyncio
async def test_check_health_reports_degraded_when_ping_times_out():
    manager = make_manager()

    await manager.register_server(
        make_stdio_config(
            "server-a",
            health_check_timeout=0.01,
        )
    )

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()

    async def hanging_ping():
        await asyncio.Event().wait()

    client.send_ping = hanging_ping

    await manager.connect_server("server-a")

    result = await manager.check_health("server-a")

    assert result.server_name == "server-a"
    assert result.status.value == "degraded"
    assert result.latency_ms >= 0.0
    assert result.error == "MCP health check timed out after 0.01 seconds."


@pytest.mark.asyncio
async def test_check_health_reports_degraded_when_ping_fails():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()
    client.send_ping = AsyncMock(
        side_effect=RuntimeError("MCP server unavailable"),
    )

    await manager.connect_server("server-a")

    result = await manager.check_health("server-a")

    assert result.server_name == "server-a"
    assert result.status.value == "degraded"
    assert result.latency_ms >= 0.0
    assert result.error == "RuntimeError: MCP server unavailable"


@pytest.mark.asyncio
async def test_check_health_tracks_consecutive_failures_and_recovery():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()
    client.send_ping = AsyncMock(
        side_effect=RuntimeError("temporary failure"),
    )

    await manager.connect_server("server-a")

    first = await manager.check_health("server-a")
    second = await manager.check_health("server-a")

    assert first.status.value == "degraded"
    assert second.status.value == "unhealthy"

    client.send_ping.side_effect = None

    third = await manager.check_health("server-a")
    fourth = await manager.check_health("server-a")

    assert third.status.value == "degraded"
    assert fourth.status.value == "healthy"

    history = manager._servers["server-a"].health_history

    assert history.consecutive_failures == 0
    assert history.consecutive_successes == 2
    assert history.last_error is None
    assert history.last_healthy_at is not None
    assert history.last_unhealthy_at is not None


@pytest.mark.asyncio
async def test_check_health_does_not_reconnect_or_discover():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    client = await manager.get_client("server-a")
    client.connect = AsyncMock()
    client.send_ping = AsyncMock()
    client.list_tools = AsyncMock()

    await manager.connect_server("server-a")
    await manager.check_health("server-a")

    client.connect.assert_awaited_once()
    client.send_ping.assert_awaited_once()
    client.list_tools.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_health_reports_unsupported_ping_as_unhealthy():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))

    class ConnectedClient:
        async def connect(self):
            return None

    manager._servers["server-a"].client = ConnectedClient()
    manager._servers["server-a"].connected = True

    result = await manager.check_health("server-a")

    assert result.status.value == "unhealthy"
    assert result.latency_ms == 0.0
    assert result.error == "MCP client does not support protocol ping."


@pytest.mark.asyncio
async def test_check_all_health_checks_every_registered_server():
    manager = make_manager()

    await manager.register_server(make_stdio_config("server-a"))
    await manager.register_server(make_stdio_config("server-b"))

    client_a = await manager.get_client("server-a")
    client_b = await manager.get_client("server-b")

    client_a.connect = AsyncMock()
    client_b.connect = AsyncMock()
    client_a.send_ping = AsyncMock()
    client_b.send_ping = AsyncMock(
        side_effect=RuntimeError("server-b unavailable"),
    )

    await manager.connect_server("server-a")
    await manager.connect_server("server-b")

    results = await manager.check_all_health()

    assert list(results) == ["server-a", "server-b"]
    assert results["server-a"].status.value == "healthy"
    assert results["server-b"].status.value == "degraded"
    client_a.send_ping.assert_awaited_once()
    client_b.send_ping.assert_awaited_once()


@pytest.mark.asyncio
async def test_recover_server_reconnects_verifies_and_discovers():
    registry = InMemoryToolRegistry()
    manager = MCPServerManager(registry)

    config = make_stdio_config(
        "server-a",
        recovery_policy=MCPRecoveryPolicy(
            max_attempts=2,
            initial_backoff=0.0,
            max_backoff=0.0,
            cooldown=0.0,
        ),
    )

    await manager.register_server(config)

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.send_ping = AsyncMock()
    client.list_tools = AsyncMock(
        return_value=[
            make_mcp_tool(name="search"),
        ]
    )

    await manager.connect_server("server-a")

    definitions = await manager.recover_server("server-a")

    assert definitions == [
        ToolDefinition(
            name="search",
            description="Search documents",
            input_schema={"type": "object"},
            metadata={
                "source": "mcp",
                "mcp_server": "server-a",
                "capability": "unclassified",
                "risk_tier": "unknown",
                "side_effect": True,
            },
        )
    ]

    client.disconnect.assert_awaited_once()
    assert client.connect.await_count == 2
    client.send_ping.assert_awaited_once()
    client.list_tools.assert_awaited_once()
    assert manager.is_connected("server-a") is True


@pytest.mark.asyncio
async def test_recover_server_retries_after_failure():
    manager = make_manager()

    await manager.register_server(
        make_stdio_config(
            "server-a",
            recovery_policy=MCPRecoveryPolicy(
                max_attempts=2,
                initial_backoff=0.0,
                max_backoff=0.0,
                cooldown=0.0,
            ),
        )
    )

    client = await manager.get_client("server-a")

    client.connect = AsyncMock(
        side_effect=[
            RuntimeError("first connection failed"),
            None,
        ]
    )
    client.disconnect = AsyncMock()
    client.send_ping = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])

    definitions = await manager.recover_server("server-a")

    assert definitions == []
    assert client.connect.await_count == 2
    assert client.send_ping.await_count == 1
    assert manager.is_connected("server-a") is True


@pytest.mark.asyncio
async def test_recover_server_exhausts_attempt_budget():
    manager = make_manager()

    await manager.register_server(
        make_stdio_config(
            "server-a",
            recovery_policy=MCPRecoveryPolicy(
                max_attempts=2,
                initial_backoff=0.0,
                max_backoff=0.0,
                cooldown=0.0,
            ),
        )
    )

    client = await manager.get_client("server-a")

    client.connect = AsyncMock(side_effect=RuntimeError("connection failed"))
    client.disconnect = AsyncMock()

    with pytest.raises(
        RuntimeError,
        match="recovery failed after 2 attempt",
    ):
        await manager.recover_server("server-a")

    assert client.connect.await_count == 2
    assert manager.is_connected("server-a") is False


@pytest.mark.asyncio
async def test_recover_server_enforces_cooldown():
    manager = make_manager()

    await manager.register_server(
        make_stdio_config(
            "server-a",
            recovery_policy=MCPRecoveryPolicy(
                max_attempts=1,
                initial_backoff=0.0,
                max_backoff=0.0,
                cooldown=60.0,
            ),
        )
    )

    client = await manager.get_client("server-a")

    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.send_ping = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])

    await manager.recover_server("server-a")

    with pytest.raises(
        RuntimeError,
        match="recovery is in cooldown",
    ):
        await manager.recover_server("server-a")

    assert client.connect.await_count == 1


@pytest.mark.asyncio
async def test_recover_server_serializes_concurrent_recovery_calls():
    manager = make_manager()

    await manager.register_server(
        make_stdio_config(
            "server-a",
            recovery_policy=MCPRecoveryPolicy(
                max_attempts=1,
                initial_backoff=0.0,
                max_backoff=0.0,
                cooldown=0.0,
            ),
        )
    )

    client = await manager.get_client("server-a")

    active = 0
    max_active = 0

    async def connect() -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1

    async def disconnect() -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1

    client.connect = AsyncMock(side_effect=connect)
    client.disconnect = AsyncMock(side_effect=disconnect)
    client.send_ping = AsyncMock()
    client.list_tools = AsyncMock(return_value=[])

    await asyncio.gather(
        manager.recover_server("server-a"),
        manager.recover_server("server-a"),
    )

    assert max_active == 1
    assert client.connect.await_count == 2
    assert client.disconnect.await_count >= 1


@pytest.mark.asyncio
async def test_recover_server_serializes_until_discovery_completes():
    manager = make_manager()

    await manager.register_server(
        make_stdio_config(
            "server-a",
            recovery_policy=MCPRecoveryPolicy(
                max_attempts=1,
                initial_backoff=0.0,
                max_backoff=0.0,
                cooldown=0.0,
            ),
        )
    )

    client = await manager.get_client("server-a")

    discovery_started = asyncio.Event()
    release_discovery = asyncio.Event()
    recovery_b_started = asyncio.Event()

    discovery_calls = 0

    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.send_ping = AsyncMock()

    async def list_tools():
        nonlocal discovery_calls

        discovery_calls += 1

        if discovery_calls == 1:
            discovery_started.set()
            await release_discovery.wait()
        else:
            recovery_b_started.set()

        return []

    client.list_tools = AsyncMock(side_effect=list_tools)

    first_recovery = asyncio.create_task(manager.recover_server("server-a"))

    await discovery_started.wait()

    second_recovery = asyncio.create_task(manager.recover_server("server-a"))

    await asyncio.sleep(0)

    assert not recovery_b_started.is_set()

    release_discovery.set()

    await asyncio.gather(
        first_recovery,
        second_recovery,
    )

    assert discovery_calls == 2
    assert recovery_b_started.is_set()
