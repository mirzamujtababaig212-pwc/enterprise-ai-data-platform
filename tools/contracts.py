from __future__ import annotations

from typing import Any, Protocol

from tools.authorization.models import (
    ToolAuthorizationRequest,
    ToolAuthorizationResult,
)
from tools.models import ToolDefinition


class Tool(Protocol):
    @property
    def definition(self) -> ToolDefinition: ...

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> Any: ...


class ToolRegistry(Protocol):
    async def register(
        self,
        tool: Tool,
    ) -> None: ...

    async def get(
        self,
        name: str,
    ) -> Tool | None: ...

    async def list_tools(
        self,
    ) -> list[ToolDefinition]: ...

    async def remove(
        self,
        name: str,
    ) -> None: ...


class ToolAuthorizer(Protocol):
    async def authorize(
        self,
        request: ToolAuthorizationRequest,
    ) -> ToolAuthorizationResult: ...
