from __future__ import annotations

from typing import Any

from ai_platform.agents.models import AgentDefinition
from tools.contracts import Tool, ToolRegistry
from tools.execution.context import ToolExecutionContext
from tools.execution.service import ToolExecutionService
from tools.mcp.resolver import (
    MCPCapabilityNotFoundError,
    MCPCapabilityResolver,
)
from tools.models import ToolDefinition


class AgentToolContext:
    """
    Controlled tool capability context for an agent.

    The context exposes only the tools explicitly declared by the
    AgentDefinition.

    Declared identifiers may be either:
    - an exact registered tool name, or
    - an MCP capability name resolved through MCPCapabilityResolver.

    Tool lookup remains owned by ToolRegistry.

    Tool execution remains owned by ToolExecutionService so that
    authorization, timeout handling, disabled-tool checks, and execution
    error handling remain centralized.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        definition: AgentDefinition,
        *,
        execution_service: ToolExecutionService | None = None,
    ) -> None:
        self._registry = registry
        self._definition = definition
        self._resolver = MCPCapabilityResolver(registry)
        self._execution_service = (
            execution_service if execution_service is not None else ToolExecutionService(registry)
        )

    @property
    def agent_name(self) -> str:
        return self._definition.name

    @property
    def tool_names(self) -> tuple[str, ...]:
        return self._definition.tool_names

    async def _resolve_declared_tools(self) -> list[ToolDefinition]:
        """
        Resolve currently available tools for the agent.

        Exact declared tool names that are missing or disabled are simply
        unavailable and therefore omitted. Capability ambiguity remains a
        configuration error and is allowed to propagate.
        """
        resolved: list[ToolDefinition] = []

        for requested_name in self._definition.tool_names:
            # Preserve the existing exact-name contract. In particular,
            # a missing or disabled exact tool must not make the entire
            # context resolution fail.
            tool = await self._registry.get(requested_name)

            if tool is not None:
                definition = tool.definition
                if definition.enabled:
                    resolved.append(definition)
                continue

            # If there is no exact tool, the declaration may instead be
            # an MCP capability identifier.
            try:
                definitions = await self._resolver.resolve_tools_for_agent(
                    (requested_name,),
                )
            except MCPCapabilityNotFoundError:
                # Missing declared identifiers are unavailable rather than
                # fatal for list/get semantics.
                continue

            for definition in definitions:
                if definition.name not in {item.name for item in resolved}:
                    resolved.append(definition)

        return resolved

    async def _resolve_declared_tool(
        self,
        name: str,
    ) -> ToolDefinition | None:
        """
        Resolve one declared tool/capability.

        Exact names take precedence. Missing or disabled exact tools return
        None. Capability ambiguity remains an error.
        """
        if not name.strip():
            raise ValueError("Tool name must not be empty.")

        if name not in self._definition.tool_names:
            raise ValueError(
                f"Tool '{name}' is not declared for agent " f"'{self._definition.name}'."
            )

        # Exact tool-name declaration has priority and must preserve the
        # historical get_tool() semantics.
        tool = await self._registry.get(name)

        if tool is not None:
            definition = tool.definition
            if not definition.enabled:
                return None
            return definition

        # No exact tool exists. The declaration may be a capability.
        try:
            definitions = await self._resolver.resolve_tools_for_agent(
                (name,),
            )
        except MCPCapabilityNotFoundError:
            return None

        return definitions[0] if definitions else None

    async def get_tool(self, name: str) -> Tool | None:
        if not name.strip():
            raise ValueError("Tool name must not be empty.")

        if name not in self._definition.tool_names:
            raise ValueError(
                f"Tool '{name}' is not declared for agent " f"'{self._definition.name}'."
            )

        definition = await self._resolve_declared_tool(name)

        if definition is None:
            return None

        return await self._registry.get(definition.name)

    async def list_tools(self) -> list[ToolDefinition]:
        return await self._resolve_declared_tools()

    async def _resolve_execution_name(self, name: str) -> str:
        """
        Resolve an execution request while preserving exact-name execution
        semantics.

        An exact declared tool name is passed directly to the execution
        service even when it is missing or disabled. This is intentional:
        ToolExecutionService owns the canonical TOOL_NOT_FOUND and
        TOOL_DISABLED failure behavior.

        If the requested name is not an exact declaration, it may be an
        actual tool resolved from one of the agent's declared capabilities.
        """
        if not name.strip():
            raise ValueError("Tool name must not be empty.")

        if name in self._definition.tool_names:
            return name

        resolved = await self._resolve_declared_tools()

        if any(definition.name == name for definition in resolved):
            return name

        raise ValueError(f"Tool '{name}' is not declared for agent " f"'{self._definition.name}'.")

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        principal: str | None = None,
        timeout_seconds: float | None = None,
        execution_context: ToolExecutionContext | None = None,
        step_id: str | None = None,
    ):
        execution_name = await self._resolve_execution_name(name)

        execution_kwargs = {
            "principal": principal,
            "timeout_seconds": timeout_seconds,
            "execution_context": execution_context,
        }

        if step_id is not None:
            execution_kwargs["step_id"] = step_id

        return await self._execution_service.execute(
            execution_name,
            arguments,
            **execution_kwargs,
        )
