from unittest.mock import AsyncMock

import pytest

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
    }

    assert definitions[1].name == "get_document"
    assert definitions[1].metadata == {
        "source": "mcp",
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
