from __future__ import annotations

from ai_platform.agents.evaluation.answer_evaluation import (
    AgentAnswerEvaluation,
    AgentAnswerEvaluator,
)
from ai_platform.agents.evaluation.models import (
    AgentContextQualityAssessment,
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
        *,
        answer_evaluation: AgentAnswerEvaluation | None = None,
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
            retrieval_score_min=evidence.retrieval_score_min,
            retrieval_score_max=evidence.retrieval_score_max,
            retrieval_score_avg=evidence.retrieval_score_avg,
            reranker_score_min=evidence.reranker_score_min,
            reranker_score_max=evidence.reranker_score_max,
            reranker_score_avg=evidence.reranker_score_avg,
            task_completed=evidence.status == "completed",
        )

        quality_gate = AgentQualityGateEvaluator.evaluate(
            metrics,
            policy,
            answer_evaluation=answer_evaluation,
        )

        return metrics, quality_gate

    @staticmethod
    def evaluate_context(
        evidence: AgentRunEvidence,
    ) -> AgentContextQualityAssessment:
        """Assess deterministic context-quality signals without changing run evaluation."""

        source_counts = dict(evidence.context_source_counts or {})
        has_context_evidence = evidence.context_assembly_events_total > 0
        has_semantic_memory_sources = (
            None if not has_context_evidence else source_counts.get("semantic_memory", 0) > 0
        )
        has_episodic_memory_sources = (
            None if not has_context_evidence else source_counts.get("episodic_memory", 0) > 0
        )
        retrieval_evidence_present = (
            None
            if not has_context_evidence
            else has_semantic_memory_sources or has_episodic_memory_sources
        )

        return AgentContextQualityAssessment(
            budget_compliant=(
                None if not has_context_evidence else not evidence.context_budget_exceeded
            ),
            retrieval_evidence_present=retrieval_evidence_present,
            has_semantic_memory_sources=has_semantic_memory_sources,
            has_episodic_memory_sources=has_episodic_memory_sources,
            assemblies_total=evidence.context_assembly_events_total,
            messages_total=evidence.context_messages_total,
            estimated_tokens_total=evidence.context_estimated_tokens_total,
            estimated_tokens_max=evidence.context_estimated_tokens_max,
            source_counts=source_counts,
        )

    @staticmethod
    def evaluate_answer(
        evidence: AgentRunEvidence,
        *,
        expected_answer: str | None,
    ) -> AgentAnswerEvaluation:
        """Evaluate the extracted final answer without changing quality-gate semantics."""
        return AgentAnswerEvaluator.evaluate(
            actual_answer=evidence.final_answer_text,
            expected_answer=expected_answer,
        )
