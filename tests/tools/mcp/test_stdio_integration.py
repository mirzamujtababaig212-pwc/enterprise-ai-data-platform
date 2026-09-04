from pathlib import Path
import sys

import pytest
from mcp import StdioServerParameters

from tools.mcp.adapter import MCPToolAdapter
from tools.mcp.sdk_client import MCPPythonSDKClient


SERVER_PATH = Path(__file__).parent / "fixtures" / "test_server.py"


@pytest.mark.asyncio
async def test_real_mcp_stdio_tool_discovery_and_execution():
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            str(SERVER_PATH),
        ],
    )

    client = MCPPythonSDKClient(server_parameters)

    try:
        await client.connect()

        tools = await client.list_tools()

        assert len(tools) == 1

        definition = tools[0]

        assert definition.name == ("search_documents")

        assert definition.description.strip() == (
            "Search a deterministic test document collection."
        )

        assert definition.input_schema["type"] == ("object")

        assert "query" in (definition.input_schema["properties"])

        adapter = MCPToolAdapter(
            client,
            definition,
        )

        result = await adapter.execute(
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
