from __future__ import annotations

from typing import Any

from ai_platform.agents.models import AgentDefinition
from tools.execution.service import ToolExecutionService
from tools.contracts import Tool, ToolRegistry
from tools.models import ToolDefinition


class AgentToolContext:
    """
    Controlled tool capability context for an agent.

    The context exposes only the tools explicitly declared by the
    AgentDefinition.

    Tool lookup remains owned by ToolRegistry.

    Tool execution remains owned by ToolExecutionService so that
    authorization, timeout handling, disabled-tool checks, and
    execution error handling remain centralized.
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

        self._execution_service = (
            execution_service if execution_service is not None else ToolExecutionService(registry)
        )

    @property
    def agent_name(self) -> str:
        """
        Return the name of the owning agent.
        """
        return self._definition.name

    @property
    def tool_names(self) -> tuple[str, ...]:
        """
        Return the tool names explicitly declared by the agent.
        """
        return self._definition.tool_names

    async def get_tool(
        self,
        name: str,
    ) -> Tool | None:
        """
        Resolve a tool if it is explicitly declared by the agent.

        Returns None when the declared tool is not registered.

        Raises:
            ValueError:
                If the tool name is empty or the tool is not declared
                by the agent.
        """
        if not name.strip():
            raise ValueError("Tool name must not be empty.")

        if name not in self._definition.tool_names:
            raise ValueError(
                f"Tool '{name}' is not declared for agent " f"'{self._definition.name}'."
            )

        return await self._registry.get(name)

    async def list_tools(self) -> list[ToolDefinition]:
        """
        Return enabled tool definitions available to the agent.

        Only tools explicitly declared in AgentDefinition.tool_names
        are returned.
        """
        definitions = await self._registry.list_tools()

        allowed_names = set(
            self._definition.tool_names,
        )

        return [definition for definition in definitions if definition.name in allowed_names]

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        principal: str | None = None,
        timeout_seconds: float | None = None,
    ):
        """
        Execute an agent-declared tool through ToolExecutionService.

        AgentToolContext does not execute tools directly. The existing
        ToolExecutionService remains the centralized execution boundary
        for authorization, timeouts, disabled tools, and failures.

        Raises:
            ValueError:
                If the tool is not declared by the agent.

        Returns:
            ToolExecutionResult:
                The normalized result returned by ToolExecutionService.
        """
        if not name.strip():
            raise ValueError("Tool name must not be empty.")

        if name not in self._definition.tool_names:
            raise ValueError(
                f"Tool '{name}' is not declared for agent " f"'{self._definition.name}'."
            )

        return await self._execution_service.execute(
            name,
            arguments,
            principal=principal,
            timeout_seconds=timeout_seconds,
        )
