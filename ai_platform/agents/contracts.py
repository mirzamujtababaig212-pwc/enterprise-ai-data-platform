from __future__ import annotations

from typing import Protocol

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import (
    AgentDefinition,
    AgentResponse,
)


class Agent(Protocol):
    """
    Contract implemented by an executable enterprise AI agent.
    """

    @property
    def definition(self) -> AgentDefinition:
        """
        Return the static definition of this agent.
        """
        ...

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        """
        Execute the agent using its request and runtime capabilities.
        """
        ...


class AgentRegistry(Protocol):
    """
    Registry contract for managing available agents.
    """

    async def register(
        self,
        agent: Agent,
    ) -> None:
        """
        Register an agent.
        """
        ...

    async def get(
        self,
        name: str,
    ) -> Agent | None:
        """
        Retrieve an agent by name.
        """
        ...

    async def list_agents(
        self,
    ) -> list[AgentDefinition]:
        """
        Return definitions for enabled agents.
        """
        ...

    async def remove(
        self,
        name: str,
    ) -> None:
        """
        Remove an agent from the registry.
        """
        ...
