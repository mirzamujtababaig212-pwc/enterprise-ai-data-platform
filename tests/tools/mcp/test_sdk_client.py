from unittest.mock import AsyncMock

import pytest

from tools.mcp.models import MCPToolCallResult
from tools.mcp.sdk_client import MCPPythonSDKClient


class FakeTool:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
    ):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class FakeListResult:
    def __init__(self):
        self.tools = [
            FakeTool(
                name="search_documents",
                description="Search enterprise documents.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                        }
                    },
                    "required": ["query"],
                },
            )
        ]


class FakeContentBlock:
    def __init__(self, text: str):
        self.text = text


class FakeCallResult:
    def __init__(
        self,
        *,
        is_error: bool = False,
        structured_content=None,
        content=None,
    ):
        self.isError = is_error
        self.structuredContent = structured_content
        self.content = content or []


class FakeSession:
    def __init__(self):
        self.list_tools = AsyncMock(return_value=FakeListResult())

        self.call_tool = AsyncMock(
            return_value=FakeCallResult(
                structured_content={
                    "results": [
                        "document-1",
                        "document-2",
                    ]
                }
            )
        )


class FailingSession:
    def __init__(self):
        self.call_tool = AsyncMock(
            return_value=FakeCallResult(
                is_error=True,
                content=[FakeContentBlock("Remote MCP server failed.")],
            )
        )


def create_connected_client(session):
    client = object.__new__(MCPPythonSDKClient)

    client.server = None
    client._exit_stack = None
    client._session = session
    client._connected = True

    return client


@pytest.mark.asyncio
async def test_extracts_mcp_tool_definitions():
    session = FakeSession()

    client = create_connected_client(session)

    tools = await client.list_tools()

    assert len(tools) == 1

    assert tools[0].name == ("search_documents")

    assert tools[0].description == ("Search enterprise documents.")

    assert tools[0].input_schema == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
            }
        },
        "required": ["query"],
    }

    session.list_tools.assert_awaited_once()


@pytest.mark.asyncio
async def test_calls_mcp_tool():
    session = FakeSession()

    client = create_connected_client(session)

    result = await client.call_tool(
        "search_documents",
        {
            "query": "enterprise AI",
        },
    )

    assert isinstance(
        result,
        MCPToolCallResult,
    )

    assert result.success is True

    assert result.output == {
        "results": [
            "document-1",
            "document-2",
        ]
    }

    session.call_tool.assert_awaited_once_with(
        "search_documents",
        arguments={
            "query": "enterprise AI",
        },
    )


@pytest.mark.asyncio
async def test_propagates_mcp_error():
    session = FailingSession()

    client = create_connected_client(session)

    result = await client.call_tool(
        "search_documents",
        {
            "query": "test",
        },
    )

    assert result.success is False

    assert result.error == ("Remote MCP server failed.")


@pytest.mark.asyncio
async def test_requires_connection_for_list_tools():
    client = object.__new__(MCPPythonSDKClient)

    client.server = None
    client._exit_stack = None
    client._session = None
    client._connected = False

    with pytest.raises(
        RuntimeError,
        match="MCP client is not connected",
    ):
        await client.list_tools()


@pytest.mark.asyncio
async def test_requires_connection_for_call_tool():
    client = object.__new__(MCPPythonSDKClient)

    client.server = None
    client._exit_stack = None
    client._session = None
    client._connected = False

    with pytest.raises(
        RuntimeError,
        match="MCP client is not connected",
    ):
        await client.call_tool(
            "search_documents",
            {},
        )


@pytest.mark.asyncio
async def test_rejects_empty_tool_name():
    session = FakeSession()

    client = create_connected_client(session)

    with pytest.raises(
        ValueError,
        match="MCP tool name must not be empty",
    ):
        await client.call_tool(
            "",
            {},
        )
