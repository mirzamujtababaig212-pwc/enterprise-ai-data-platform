from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from ai_platform.agents.orchestration import AgentRuntimeDecision
from ai_platform.agents.runtime_evaluation import AgentRuntimeEvaluationSnapshot

if TYPE_CHECKING:
    from ai_platform.agents.execution import AgentExecutionContext


class AgentRuntimeDecisionReason(StrEnum):
    """Deterministic reason explaining a Runtime V1 decision."""

    RESPONSE_OUTPUT_MISSING = "response_output_missing"
    RESPONSE_OUTPUT_BLANK = "response_output_blank"
    ITERATION_BUDGET_REMAINING = "iteration_budget_remaining"
    ITERATION_BUDGET_EXHAUSTED = "iteration_budget_exhausted"


@dataclass(frozen=True)
class AgentRuntimeDecisionResult:
    """Immutable result produced by a Runtime V1 decision provider."""

    decision: AgentRuntimeDecision
    reason: AgentRuntimeDecisionReason

    def __post_init__(self) -> None:
        if not isinstance(self.decision, AgentRuntimeDecision):
            raise TypeError("Runtime decision result decision must be an AgentRuntimeDecision.")

        if not isinstance(self.reason, AgentRuntimeDecisionReason):
            raise TypeError("Runtime decision result reason must be an AgentRuntimeDecisionReason.")


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
    ) -> AgentRuntimeDecisionResult:
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
    ) -> AgentRuntimeDecisionResult:
        """Evaluate response validity before applying the iteration bound."""
        if not hasattr(context, "agent_name"):
            raise TypeError("Agent runtime decision provider requires an agent execution context.")

        if evaluation.response.output is None:
            return AgentRuntimeDecisionResult(
                decision=AgentRuntimeDecision.STOP,
                reason=AgentRuntimeDecisionReason.RESPONSE_OUTPUT_MISSING,
            )

        if isinstance(evaluation.response.output, str) and not evaluation.response.output.strip():
            return AgentRuntimeDecisionResult(
                decision=AgentRuntimeDecision.STOP,
                reason=AgentRuntimeDecisionReason.RESPONSE_OUTPUT_BLANK,
            )

        if evaluation.iteration < context.execution_budget.max_iterations:
            return AgentRuntimeDecisionResult(
                decision=AgentRuntimeDecision.CONTINUE,
                reason=AgentRuntimeDecisionReason.ITERATION_BUDGET_REMAINING,
            )

        return AgentRuntimeDecisionResult(
            decision=AgentRuntimeDecision.STOP,
            reason=AgentRuntimeDecisionReason.ITERATION_BUDGET_EXHAUSTED,
        )
