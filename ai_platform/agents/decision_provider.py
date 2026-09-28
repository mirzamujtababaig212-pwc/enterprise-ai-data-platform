from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ai_platform.agents.orchestration import AgentRuntimeDecision
from ai_platform.agents.runtime_evaluation import AgentRuntimeEvaluationSnapshot

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
        evaluation: AgentRuntimeEvaluationSnapshot,
    ) -> AgentRuntimeDecision:
        """Produce the runtime decision for the completed iteration."""
        ...


class DeterministicAgentRuntimeDecisionProvider:
    """
    Default runtime decision provider.

    Runtime V1 preserves single-iteration behavior by default while
    allowing bounded semantic replanning through the execution budget.
    """

    def evaluate(
        self,
        context: AgentExecutionContext,
        evaluation: AgentRuntimeEvaluationSnapshot,
    ) -> AgentRuntimeDecision:
        """Evaluate response validity before applying the iteration bound."""
        if not hasattr(context, "agent_name"):
            raise TypeError("Agent runtime decision provider requires an agent execution context.")

        if evaluation.response.output is None:
            return AgentRuntimeDecision.STOP

        if isinstance(evaluation.response.output, str) and not evaluation.response.output.strip():
            return AgentRuntimeDecision.STOP

        if evaluation.iteration < context.execution_budget.max_iterations:
            return AgentRuntimeDecision.CONTINUE

        return AgentRuntimeDecision.STOP
