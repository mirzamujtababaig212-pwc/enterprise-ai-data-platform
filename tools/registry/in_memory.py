from __future__ import annotations

from tools.contracts import Tool
from tools.models import ToolDefinition


class InMemoryToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    async def register(self, tool: Tool) -> None:
        definition = tool.definition

        if not definition.name.strip():
            raise ValueError("Tool name must not be empty.")

        if not definition.description.strip():
            raise ValueError("Tool description must not be empty.")

        existing = self._tools.get(definition.name)

        if existing is not None:
            existing_metadata = existing.definition.metadata
            new_metadata = definition.metadata

            existing_mcp_server = existing_metadata.get("mcp_server")
            new_mcp_server = new_metadata.get("mcp_server")

            if (
                existing_mcp_server is not None
                and new_mcp_server is not None
                and existing_mcp_server != new_mcp_server
            ):
                raise ValueError(
                    f"Tool '{definition.name}' is already registered "
                    f"by MCP server '{existing_mcp_server}' and cannot "
                    f"also be registered by MCP server '{new_mcp_server}'."
                )

        self._tools[definition.name] = tool

    async def get(self, name: str) -> Tool | None:
        if not name.strip():
            raise ValueError("Tool name must not be empty.")

        return self._tools.get(name)

    async def list_tools(self) -> list[ToolDefinition]:
        return [tool.definition for tool in self._tools.values() if tool.definition.enabled]

    async def remove(self, name: str) -> None:
        if not name.strip():
            raise ValueError("Tool name must not be empty.")

        self._tools.pop(name, None)
