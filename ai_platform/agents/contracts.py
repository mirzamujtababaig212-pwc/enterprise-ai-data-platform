from __future__ import annotations

from typing import Protocol

from ai_platform.agents.checkpoint import AgentExecutionCheckpoint
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


class RecoverableAgent(Protocol):
    """
    Optional capability implemented by agents that support durable recovery.

    Recovery is deliberately separate from the base Agent contract so that
    ordinary agents do not need to implement resume().
    """

    @property
    def definition(self) -> AgentDefinition: ...

    async def resume(
        self,
        context: AgentExecutionContext,
        checkpoint: AgentExecutionCheckpoint,
    ) -> AgentResponse: ...


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
