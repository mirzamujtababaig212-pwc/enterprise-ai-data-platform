from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ai_platform.agents.orchestration import OrchestrationPlan
from ai_platform.agents.plans import build_agent_orchestration_plan

if TYPE_CHECKING:
    from ai_platform.agents.execution import AgentExecutionContext


class AgentPlanProvider(Protocol):
    """
    Contract for producing an immutable orchestration plan.

    A provider creates a plan definition; it does not execute the plan
    and does not own mutable orchestration state.
    """

    def build_plan(
        self,
        context: AgentExecutionContext,
    ) -> OrchestrationPlan:
        """Build the orchestration plan for one semantic runtime iteration."""
        ...


class DeterministicAgentPlanProvider:
    """
    Application-owned plan provider backed by the registered plan factory.

    This provider preserves the current deterministic orchestration behavior
    while introducing the abstraction required for future replanning.
    """

    def build_plan(
        self,
        context: AgentExecutionContext,
    ) -> OrchestrationPlan:
        """Build a fresh immutable plan for the supplied execution context."""
        if not hasattr(context, "agent_name"):
            raise TypeError("Agent plan provider requires an agent execution context.")

        return build_agent_orchestration_plan(context.agent_name)
