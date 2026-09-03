from __future__ import annotations

from typing import Any, Protocol, Sequence

from tools.mcp.models import (
    MCPToolCallResult,
    MCPToolDefinition,
)


class MCPClient(Protocol):
    async def list_tools(
        self,
    ) -> Sequence[MCPToolDefinition]: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult: ...
