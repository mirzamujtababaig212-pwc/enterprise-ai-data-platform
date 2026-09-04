from __future__ import annotations

import asyncio
from typing import Any

from tools.authorization.service import ToolAuthorizationService
from tools.contracts import ToolRegistry
from tools.models import ToolExecutionResult


class ToolExecutionService:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        authorization_service: ToolAuthorizationService | None = None,
        default_timeout_seconds: float = 30.0,
    ):
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be greater than zero.")

        self.registry = registry
        self.authorization_service = authorization_service
        self.default_timeout_seconds = default_timeout_seconds

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        principal: str | None = None,
        timeout_seconds: float | None = None,
    ) -> ToolExecutionResult:
        if not tool_name.strip():
            raise ValueError("Tool name must not be empty.")

        if arguments is None:
            raise ValueError("Tool arguments must not be None.")

        timeout = self.default_timeout_seconds if timeout_seconds is None else timeout_seconds

        if timeout <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        tool = await self.registry.get(tool_name)

        if tool is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool not found: {tool_name}",
            )

        if not tool.definition.enabled:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool is disabled: {tool_name}",
            )

        if self.authorization_service is not None:
            if principal is None or not principal.strip():
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=("Principal is required when " "tool authorization is enabled."),
                )

            authorization = await self.authorization_service.authorize(
                principal,
                tool_name,
            )

            if not authorization.allowed:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=(authorization.reason or "Tool execution is not authorized."),
                )

        try:
            output = await asyncio.wait_for(
                tool.execute(arguments),
                timeout=timeout,
            )

            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                output=output,
            )

        except asyncio.TimeoutError:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=(f"Tool execution timed out after " f"{timeout} seconds: {tool_name}"),
            )

        except Exception as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
