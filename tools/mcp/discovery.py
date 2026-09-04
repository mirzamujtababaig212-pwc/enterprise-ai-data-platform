from __future__ import annotations

from tools.contracts import ToolRegistry
from tools.mcp.adapter import MCPToolAdapter
from tools.mcp.client import MCPClient
from tools.models import ToolDefinition


class MCPToolDiscoveryService:
    """
    Discovers tools from an MCP server and registers them
    with the platform's existing ToolRegistry.
    """

    def __init__(
        self,
        client: MCPClient,
        registry: ToolRegistry,
    ) -> None:
        self.client = client
        self.registry = registry

    async def discover_and_register(
        self,
    ) -> list[ToolDefinition]:
        mcp_tools = await self.client.list_tools()

        definitions: list[ToolDefinition] = []

        for mcp_tool in mcp_tools:
            adapter = MCPToolAdapter(
                self.client,
                mcp_tool,
            )

            await self.registry.register(adapter)

            definitions.append(adapter.definition)

        return definitions
