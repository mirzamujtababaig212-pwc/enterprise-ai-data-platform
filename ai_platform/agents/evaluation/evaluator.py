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
            rag_queries_total=evidence.rag_queries_total,
            rag_sources_retrieved_total=evidence.rag_sources_retrieved_total,
            has_final_answer=evidence.has_final_answer,
            final_answer_length=evidence.final_answer_length,
            rag_sources_available_count=evidence.rag_sources_available_count,
            rag_unique_chunks_count=evidence.rag_unique_chunks_count,
            task_completed=evidence.status == "completed",
        )

        quality_gate = AgentQualityGateEvaluator.evaluate(
            metrics,
            policy,
        )

        return metrics, quality_gate
