from __future__ import annotations

from typing import Any

from tools.mcp.client import MCPClient
from tools.mcp.models import MCPToolDefinition
from tools.models import ToolDefinition


class MCPToolAdapter:
    def __init__(
        self,
        client: MCPClient,
        definition: MCPToolDefinition,
        server_name: str | None = None,
    ):
        if not definition.name.strip():
            raise ValueError("MCP tool name must not be empty.")

        if not definition.description.strip():
            raise ValueError("MCP tool description must not be empty.")

        if server_name is not None and not server_name.strip():
            raise ValueError("MCP server name must not be empty.")

        self.client = client
        self._definition = definition
        self._server_name = server_name

    @property
    def definition(self) -> ToolDefinition:
        metadata = {
            "source": "mcp",
        }

        if self._server_name is not None:
            metadata["mcp_server"] = self._server_name

        return ToolDefinition(
            name=self._definition.name,
            description=self._definition.description,
            input_schema=dict(self._definition.input_schema),
            metadata=metadata,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        if arguments is None:
            raise ValueError("Tool arguments must not be None.")

        result = await self.client.call_tool(
            self._definition.name,
            arguments,
        )

        if not result.success:
            raise RuntimeError(result.error or "MCP tool execution failed.")

        return result.output
