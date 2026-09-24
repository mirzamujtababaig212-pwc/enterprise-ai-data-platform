from __future__ import annotations

from ai_platform.agents.evaluation.models import (
    AgentEvaluationMetrics,
    AgentRunEvidence,
)
from ai_platform.agents.evaluation.policy import (
    AgentEvaluationPolicy,
    AgentQualityGateEvaluator,
    AgentQualityGateResult,
)


class AgentEvaluator:
    """Deterministic evaluator operating over consolidated agent-run evidence."""

    @staticmethod
    def evaluate_run(
        evidence: AgentRunEvidence,
        policy: AgentEvaluationPolicy,
    ) -> tuple[AgentEvaluationMetrics, AgentQualityGateResult]:
        metrics = AgentEvaluationMetrics(
            execution_time_ms=evidence.execution_time_ms,
            steps_total=evidence.total_steps,
            tool_calls_total=evidence.tool_calls_total,
            tool_calls_successful=evidence.tool_calls_successful,
            tool_calls_failed=evidence.tool_calls_failed,
            invalid_tool_calls=evidence.invalid_tool_calls,
            governance_denials=evidence.governance_denials,
            task_completed=evidence.status == "completed",
        )

        quality_gate = AgentQualityGateEvaluator.evaluate(
            metrics,
            policy,
        )

        return metrics, quality_gate
