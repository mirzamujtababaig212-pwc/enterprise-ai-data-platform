from __future__ import annotations

import asyncio
import logging
from typing import Any

from tools.authorization.audit import (
    ToolAuthorizationAuditRecord,
    ToolAuthorizationAuditSink,
)
from tools.authorization.service import ToolAuthorizationService
from tools.contracts import ToolRegistry
from tools.models import ToolExecutionResult

logger = logging.getLogger(__name__)


class ToolExecutionService:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        authorization_service: ToolAuthorizationService | None = None,
        audit_sink: ToolAuthorizationAuditSink | None = None,
        default_timeout_seconds: float = 30.0,
    ):
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be greater than zero.")

        self.registry = registry
        self.authorization_service = authorization_service
        self.audit_sink = audit_sink
        self.default_timeout_seconds = default_timeout_seconds

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        principal: str | None = None,
        timeout_seconds: float | None = None,
        execution_context: dict[str, Any] | None = None,
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
                metadata=tool.definition.metadata,
            )

            await self._audit_authorization(
                principal=principal,
                tool_name=tool_name,
                allowed=authorization.allowed,
                reason=authorization.reason,
                execution_context=execution_context,
            )

            if not authorization.allowed:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=(authorization.reason or "Tool execution is not authorized."),
                )

        try:
            contextual_execute = getattr(tool, "execute_with_context", None)

            if contextual_execute is not None:
                execution = contextual_execute(
                    arguments,
                    execution_context or {},
                )
            else:
                execution = tool.execute(arguments)

            output = await asyncio.wait_for(
                execution,
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

    async def _audit_authorization(
        self,
        *,
        principal: str,
        tool_name: str,
        allowed: bool,
        reason: str | None,
        execution_context: dict[str, Any] | None,
    ) -> None:
        if self.audit_sink is None:
            return

        context = execution_context or {}

        record = ToolAuthorizationAuditRecord(
            principal=principal,
            tool_name=tool_name,
            allowed=allowed,
            reason=reason,
            run_id=context.get("run_id"),
            call_id=context.get("call_id"),
            agent_name=context.get("agent_name"),
            session_id=context.get("session_id"),
        )

        try:
            await self.audit_sink.record(record)
        except Exception:
            logger.exception(
                "Failed to record tool authorization audit: "
                "principal=%s tool_name=%s allowed=%s",
                principal,
                tool_name,
                allowed,
            )
