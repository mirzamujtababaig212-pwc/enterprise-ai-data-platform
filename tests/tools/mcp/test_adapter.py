import pytest

from tools.mcp.adapter import MCPToolAdapter
from tools.mcp.config import MCPToolCapability
from tools.mcp.models import (
    MCPToolCallResult,
    MCPToolDefinition,
)


class FakeMCPClient:
    def __init__(self):
        self.calls = []

    async def list_tools(self):
        return [
            MCPToolDefinition(
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

    async def call_tool(
        self,
        name,
        arguments,
        *,
        meta=None,
    ):
        self.calls.append(
            {
                "name": name,
                "arguments": arguments,
                "meta": meta,
            }
        )

        return MCPToolCallResult(
            output={
                "results": [
                    "document-1",
                    "document-2",
                ]
            }
        )


class FailingMCPClient:
    async def call_tool(
        self,
        name,
        arguments,
    ):
        return MCPToolCallResult(error="Remote MCP server failed.")


@pytest.mark.asyncio
async def test_mcp_adapter_exposes_internal_tool_definition():
    client = FakeMCPClient()

    definition = MCPToolDefinition(
        name="search_documents",
        description="Search enterprise documents.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                }
            },
        },
    )

    tool = MCPToolAdapter(
        client,
        definition,
    )

    assert tool.definition.name == ("search_documents")

    assert tool.definition.description == ("Search enterprise documents.")

    assert tool.definition.input_schema == {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
            }
        },
    }

    assert tool.definition.metadata == {
        "source": "mcp",
    }


@pytest.mark.asyncio
async def test_mcp_adapter_executes_remote_tool():
    client = FakeMCPClient()

    definition = MCPToolDefinition(
        name="search_documents",
        description="Search enterprise documents.",
    )

    tool = MCPToolAdapter(
        client,
        definition,
    )

    result = await tool.execute(
        {
            "query": "enterprise AI",
        }
    )

    assert result == {
        "results": [
            "document-1",
            "document-2",
        ]
    }

    assert client.calls == [
        {
            "name": "search_documents",
            "arguments": {
                "query": "enterprise AI",
            },
            "meta": None,
        }
    ]


@pytest.mark.asyncio
async def test_mcp_adapter_propagates_remote_failure():
    client = FailingMCPClient()

    definition = MCPToolDefinition(
        name="failing_tool",
        description="A failing MCP tool.",
    )

    tool = MCPToolAdapter(
        client,
        definition,
    )

    with pytest.raises(
        RuntimeError,
        match="Remote MCP server failed",
    ):
        await tool.execute({})


@pytest.mark.asyncio
async def test_mcp_adapter_rejects_none_arguments():
    client = FakeMCPClient()

    definition = MCPToolDefinition(
        name="search_documents",
        description="Search enterprise documents.",
    )

    tool = MCPToolAdapter(
        client,
        definition,
    )

    with pytest.raises(
        ValueError,
        match="Tool arguments must not be None",
    ):
        await tool.execute(None)


def test_mcp_adapter_rejects_empty_name():
    client = FakeMCPClient()

    definition = MCPToolDefinition(
        name="",
        description="Invalid MCP tool.",
    )

    with pytest.raises(
        ValueError,
        match="MCP tool name must not be empty",
    ):
        MCPToolAdapter(
            client,
            definition,
        )


def test_mcp_adapter_rejects_empty_description():
    client = FakeMCPClient()

    definition = MCPToolDefinition(
        name="invalid_tool",
        description="",
    )

    with pytest.raises(
        ValueError,
        match="MCP tool description must not be empty",
    ):
        MCPToolAdapter(
            client,
            definition,
        )


@pytest.mark.asyncio
async def test_mcp_adapter_forwards_external_idempotency_key_in_meta():
    from tools.execution.context import ToolExecutionContext

    client = FakeMCPClient()

    definition = MCPToolDefinition(
        name="send_payment",
        description="Send a payment.",
    )

    tool = MCPToolAdapter(
        client,
        definition,
    )

    context = ToolExecutionContext(
        run_id="run-123",
        call_id="call-456",
    )

    result = await tool.execute_with_context(
        {
            "amount": 100,
        },
        context,
    )

    assert result == {
        "results": [
            "document-1",
            "document-2",
        ]
    }

    assert client.calls == [
        {
            "name": "send_payment",
            "arguments": {
                "amount": 100,
            },
            "meta": {
                "deldai": {
                    "idempotency_key": "deldai:run-123:call-456:send_payment",
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_mcp_adapter_does_not_add_meta_without_execution_identity():
    client = FakeMCPClient()

    definition = MCPToolDefinition(
        name="search_documents",
        description="Search enterprise documents.",
    )

    tool = MCPToolAdapter(
        client,
        definition,
    )

    await tool.execute_with_context(
        {
            "query": "enterprise AI",
        },
        None,
    )

    assert client.calls == [
        {
            "name": "search_documents",
            "arguments": {
                "query": "enterprise AI",
            },
            "meta": None,
        }
    ]


@pytest.mark.asyncio
async def test_mcp_adapter_exposes_deldai_capability_metadata():
    client = FakeMCPClient()

    definition = MCPToolDefinition(
        name="search_documents",
        description="Search enterprise documents.",
    )

    tool = MCPToolAdapter(
        client,
        definition,
        server_name="document-server",
        capability=MCPToolCapability(
            capability="document.read",
            risk_tier="low",
            side_effect=False,
            permission_scope="document:read",
        ),
    )

    assert tool.definition.metadata == {
        "source": "mcp",
        "mcp_server": "document-server",
        "capability": "document.read",
        "risk_tier": "low",
        "side_effect": False,
        "permission_scope": "document:read",
    }
