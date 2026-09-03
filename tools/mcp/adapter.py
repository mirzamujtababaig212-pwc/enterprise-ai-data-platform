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
    ):
        if not definition.name.strip():
            raise ValueError("MCP tool name must not be empty.")

        if not definition.description.strip():
            raise ValueError("MCP tool description must not be empty.")

        self.client = client
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._definition.name,
            description=self._definition.description,
            input_schema=dict(self._definition.input_schema),
            metadata={
                "source": "mcp",
            },
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
