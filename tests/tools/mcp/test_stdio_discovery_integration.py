from pathlib import Path
import sys

import pytest
from mcp import StdioServerParameters

from tools.mcp.discovery import MCPToolDiscoveryService
from tools.mcp.sdk_client import MCPPythonSDKClient
from tools.registry.in_memory import InMemoryToolRegistry


SERVER_PATH = Path(__file__).parent / "fixtures" / "test_server.py"


@pytest.mark.asyncio
async def test_real_mcp_stdio_discovery_registers_tools():
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            str(SERVER_PATH),
        ],
    )

    client = MCPPythonSDKClient(server_parameters)

    registry = InMemoryToolRegistry()

    service = MCPToolDiscoveryService(
        client=client,
        registry=registry,
    )

    try:
        await client.connect()

        definitions = await service.discover_and_register()

        assert len(definitions) == 1

        definition = definitions[0]

        assert definition.name == ("search_documents")

        assert definition.description.strip() == (
            "Search a deterministic test document collection."
        )

        assert definition.metadata == {
            "source": "mcp",
        }

        registered_tools = await registry.list_tools()

        assert len(registered_tools) == 1

        registered_definition = registered_tools[0]

        assert registered_definition.name == ("search_documents")

        assert registered_definition.metadata == {
            "source": "mcp",
        }

        tool = await registry.get("search_documents")

        assert tool is not None

        result = await tool.execute(
            {
                "query": "enterprise AI",
            }
        )

        assert result["query"] == ("enterprise AI")

        assert result["results"] == [
            {
                "id": "document-1",
                "content": ("Enterprise AI platform architecture."),
            }
        ]

    finally:
        await client.disconnect()
