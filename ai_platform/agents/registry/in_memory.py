from __future__ import annotations

from ai_platform.agents.contracts import Agent
from ai_platform.agents.models import AgentDefinition


class InMemoryAgentRegistry:
    """
    In-memory implementation of the AgentRegistry contract.

    Agents are keyed by their definition name.

    This registry intentionally owns only registration and lookup.
    It does not execute agents and does not depend on the LLM Gateway,
    Tool Registry, MCP, Memory, or RAG layers.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    async def register(
        self,
        agent: Agent,
    ) -> None:
        """
        Register an agent.

        Registering another agent with the same name replaces the
        existing registration.
        """
        definition = agent.definition

        self._validate_definition(definition)

        self._agents[definition.name] = agent

    async def get(
        self,
        name: str,
    ) -> Agent | None:
        """
        Retrieve an agent by name.
        """
        if not name.strip():
            raise ValueError("Agent name must not be empty.")

        return self._agents.get(name)

    async def list_agents(
        self,
    ) -> list[AgentDefinition]:
        """
        Return definitions for enabled agents.

        Definitions are returned in registration order.
        """
        return [agent.definition for agent in self._agents.values() if agent.definition.enabled]

    async def remove(
        self,
        name: str,
    ) -> None:
        """
        Remove an agent from the registry.

        Removing a name that does not exist is a no-op.
        """
        if not name.strip():
            raise ValueError("Agent name must not be empty.")

        self._agents.pop(name, None)

    @staticmethod
    def _validate_definition(
        definition: AgentDefinition,
    ) -> None:
        """
        Validate registry-level agent requirements.
        """
        if not definition.name.strip():
            raise ValueError("Agent name must not be empty.")

        if not definition.description.strip():
            raise ValueError("Agent description must not be empty.")
