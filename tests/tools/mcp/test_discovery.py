from unittest.mock import AsyncMock

import pytest

from tools.mcp.config import MCPToolCapability
from tools.mcp.discovery import MCPToolDiscoveryService
from tools.mcp.models import MCPToolDefinition
from tools.registry.in_memory import InMemoryToolRegistry


class FakeMCPClient:
    def __init__(self) -> None:
        self.list_tools = AsyncMock(
            return_value=[
                MCPToolDefinition(
                    name="search_documents",
                    description="Search enterprise documents.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                MCPToolDefinition(
                    name="get_document",
                    description="Get an enterprise document.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "document_id": {
                                "type": "string",
                            },
                        },
                        "required": ["document_id"],
                    },
                ),
            ]
        )


@pytest.mark.asyncio
async def test_discovers_and_registers_mcp_tools():
    client = FakeMCPClient()
    registry = InMemoryToolRegistry()

    service = MCPToolDiscoveryService(
        client=client,
        registry=registry,
    )

    definitions = await service.discover_and_register()

    assert len(definitions) == 2

    assert definitions[0].name == "search_documents"
    assert definitions[0].metadata == {
        "source": "mcp",
        "capability": "unclassified",
        "risk_tier": "unknown",
        "side_effect": True,
    }

    assert definitions[1].name == "get_document"
    assert definitions[1].metadata == {
        "source": "mcp",
        "capability": "unclassified",
        "risk_tier": "unknown",
        "side_effect": True,
    }

    client.list_tools.assert_awaited_once()

    registered_tools = await registry.list_tools()

    assert len(registered_tools) == 2

    assert {definition.name for definition in registered_tools} == {
        "search_documents",
        "get_document",
    }


@pytest.mark.asyncio
async def test_discovery_with_no_mcp_tools_returns_empty_list():
    client = FakeMCPClient()
    client.list_tools = AsyncMock(return_value=[])

    registry = InMemoryToolRegistry()

    service = MCPToolDiscoveryService(
        client=client,
        registry=registry,
    )

    definitions = await service.discover_and_register()

    assert definitions == []
    assert await registry.list_tools() == []

    client.list_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_registered_mcp_tool_can_be_retrieved_from_registry():
    client = FakeMCPClient()
    registry = InMemoryToolRegistry()

    service = MCPToolDiscoveryService(
        client=client,
        registry=registry,
    )

    await service.discover_and_register()

    tool = await registry.get("search_documents")

    assert tool is not None
    assert tool.definition.name == "search_documents"
    assert tool.definition.metadata == {
        "source": "mcp",
        "capability": "unclassified",
        "risk_tier": "unknown",
        "side_effect": True,
    }


@pytest.mark.asyncio
async def test_discovery_propagates_client_failure():
    client = FakeMCPClient()
    client.list_tools = AsyncMock(side_effect=RuntimeError("MCP discovery failed."))

    registry = InMemoryToolRegistry()

    service = MCPToolDiscoveryService(
        client=client,
        registry=registry,
    )

    with pytest.raises(
        RuntimeError,
        match="MCP discovery failed.",
    ):
        await service.discover_and_register()

    assert await registry.list_tools() == []


@pytest.mark.asyncio
async def test_discovery_propagates_deldai_tool_capability_metadata():
    client = FakeMCPClient()
    registry = InMemoryToolRegistry()

    service = MCPToolDiscoveryService(
        client=client,
        registry=registry,
        server_name="document-server",
        tool_capabilities={
            "search_documents": MCPToolCapability(
                capability="document.read",
                risk_tier="low",
                side_effect=False,
                permission_scope="document:read",
            ),
        },
    )

    definitions = await service.discover_and_register()

    assert definitions[0].metadata == {
        "source": "mcp",
        "mcp_server": "document-server",
        "capability": "document.read",
        "risk_tier": "low",
        "side_effect": False,
        "permission_scope": "document:read",
    }


@pytest.mark.asyncio
async def test_discovery_assigns_conservative_capability_to_unclassified_tool():
    client = FakeMCPClient()
    client.list_tools = AsyncMock(
        return_value=[
            MCPToolDefinition(
                name="unknown_tool",
                description="An unclassified MCP tool",
                input_schema={"type": "object"},
            )
        ]
    )

    service = MCPToolDiscoveryService(
        client=client,
        registry=InMemoryToolRegistry(),
        server_name="external-server",
    )

    definitions = await service.discover_and_register()

    assert len(definitions) == 1
    assert definitions[0].metadata == {
        "source": "mcp",
        "mcp_server": "external-server",
        "capability": "unclassified",
        "risk_tier": "unknown",
        "side_effect": True,
    }


@pytest.mark.asyncio
async def test_discovery_prefers_explicit_capability_over_default():
    capability = MCPToolCapability(
        capability="document.search",
        risk_tier="low",
        side_effect=False,
        permission_scope="documents:read",
    )

    client = FakeMCPClient()
    client.list_tools = AsyncMock(
        return_value=[
            MCPToolDefinition(
                name="search_documents",
                description="Search documents",
                input_schema={"type": "object"},
            )
        ]
    )

    service = MCPToolDiscoveryService(
        client=client,
        registry=InMemoryToolRegistry(),
        server_name="document-server",
        tool_capabilities={
            "search_documents": capability,
        },
    )

    definitions = await service.discover_and_register()

    assert definitions[0].metadata == {
        "source": "mcp",
        "mcp_server": "document-server",
        "capability": "document.search",
        "risk_tier": "low",
        "side_effect": False,
        "permission_scope": "documents:read",
    }


@pytest.mark.asyncio
async def test_discovery_rejects_duplicate_tool_identity_across_mcp_servers():
    client_a = FakeMCPClient()
    client_b = FakeMCPClient()

    registry = InMemoryToolRegistry()

    service_a = MCPToolDiscoveryService(
        client=client_a,
        registry=registry,
        server_name="server-a",
    )

    service_b = MCPToolDiscoveryService(
        client=client_b,
        registry=registry,
        server_name="server-b",
    )

    await service_a.discover_and_register()

    with pytest.raises(
        ValueError,
        match="already registered",
    ):
        await service_b.discover_and_register()
