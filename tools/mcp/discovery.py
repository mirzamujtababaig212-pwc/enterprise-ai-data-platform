from __future__ import annotations

from tools.contracts import ToolRegistry
from tools.mcp.adapter import MCPToolAdapter
from tools.mcp.client import MCPClient
from tools.mcp.config import (
    MCPToolCapability,
    resolve_mcp_tool_capability,
)
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
        server_name: str | None = None,
        tool_capabilities: dict[str, MCPToolCapability] | None = None,
    ) -> None:
        if server_name is not None and not server_name.strip():
            raise ValueError("MCP server name must not be empty.")

        self.client = client
        self.registry = registry
        self.server_name = server_name
        self.tool_capabilities = dict(tool_capabilities or {})

    async def discover_and_register(
        self,
    ) -> list[ToolDefinition]:
        mcp_tools = await self.client.list_tools()

        definitions: list[ToolDefinition] = []

        for mcp_tool in mcp_tools:
            capability = resolve_mcp_tool_capability(
                self.tool_capabilities.get(mcp_tool.name),
            )

            adapter = MCPToolAdapter(
                self.client,
                mcp_tool,
                server_name=self.server_name,
                capability=capability,
            )

            await self.registry.register(adapter)

            definitions.append(adapter.definition)

        await self._reconcile_stale_tools(
            discovered_tool_names={definition.name for definition in definitions},
        )

        return definitions

    async def _reconcile_stale_tools(
        self,
        *,
        discovered_tool_names: set[str],
    ) -> None:
        if self.server_name is None:
            return

        registered_tools = await self.registry.list_tools()

        stale_tool_names = {
            definition.name
            for definition in registered_tools
            if definition.metadata.get("mcp_server") == self.server_name
            and definition.name not in discovered_tool_names
        }

        for tool_name in stale_tool_names:
            await self.registry.remove(tool_name)
