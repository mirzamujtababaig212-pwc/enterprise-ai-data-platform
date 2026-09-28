from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ai_platform.agents.orchestration import AgentRuntimeDecision

if TYPE_CHECKING:
    from ai_platform.agents.execution import AgentExecutionContext


class AgentRuntimeDecisionProvider(Protocol):
    """
    Contract for producing the runtime decision after evaluation.

    A decision provider decides whether the semantic runtime should stop
    or begin another planning iteration. It does not execute plans.
    """

    def evaluate(
        self,
        context: AgentExecutionContext,
    ) -> AgentRuntimeDecision:
        """Produce the runtime decision for the completed iteration."""
        ...


class DeterministicAgentRuntimeDecisionProvider:
    """
    Default runtime decision provider.

    Runtime V1 preserves the current single-iteration behavior by
    deterministically returning STOP.
    """

    def evaluate(
        self,
        context: AgentExecutionContext,
    ) -> AgentRuntimeDecision:
        """Stop after the current semantic runtime iteration."""
        if not hasattr(context, "agent_name"):
            raise TypeError("Runtime decision provider requires an agent execution context.")

        return AgentRuntimeDecision.STOP
