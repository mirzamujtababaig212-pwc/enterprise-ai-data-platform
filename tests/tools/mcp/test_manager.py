from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from tools.mcp.config import MCPServerConfig, MCPToolCapability
from tools.mcp.manager import MCPServerManager
from tools.mcp.models import MCPToolDefinition
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
async def test_check_health_reports_unhealthy_when_ping_times_out():
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
    assert result.status.value == "unhealthy"
    assert result.latency_ms >= 0.0
    assert result.error == "MCP health check timed out after 0.01 seconds."


@pytest.mark.asyncio
async def test_check_health_reports_unhealthy_when_ping_fails():
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
    assert result.status.value == "unhealthy"
    assert result.latency_ms >= 0.0
    assert result.error == "RuntimeError: MCP server unavailable"


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
    assert results["server-b"].status.value == "unhealthy"
    client_a.send_ping.assert_awaited_once()
    client_b.send_ping.assert_awaited_once()
