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
