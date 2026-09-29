from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class AgentRunEvidence:
    """Consolidated execution evidence extracted from an agent execution run."""

    run_id: str
    agent_name: str
    agent_version: str | None
    tenant_id: str | None
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    execution_time_ms: float
    total_steps: int
    tool_calls_total: int
    tool_calls_successful: int
    tool_calls_failed: int
    invalid_tool_calls: int
    governance_denials: int
    rag_queries_total: int = 0
    rag_sources_retrieved_total: int = 0
    has_final_answer: bool = False
    final_answer_length: int = 0
    final_answer_text: str | None = None
    has_rag_provenance: bool = False
    has_rag_sources_available: bool = False
    rag_sources_available_count: int = 0
    rag_unique_chunks_count: int = 0
    retrieval_score_min: float | None = None
    retrieval_score_max: float | None = None
    retrieval_score_avg: float | None = None
    reranker_score_min: float | None = None
    reranker_score_max: float | None = None
    reranker_score_avg: float | None = None
    effective_model: str | None = None
    effective_provider: str | None = None
    model_policy_id: str | None = None
    model_policy_version: str | None = None
    context_assembly_events_total: int = 0
    context_messages_total: int = 0
    context_estimated_tokens_total: int = 0
    context_estimated_tokens_max: int = 0
    context_estimated_remaining_after_context_min: int | None = None
    context_budget_exceeded: bool = False
    context_source_profile_changes: int = 0
    context_source_counts: dict[str, int] | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty.")

        if not self.agent_name.strip():
            raise ValueError("agent_name must not be empty.")

        if self.execution_time_ms < 0:
            raise ValueError("execution_time_ms must be non-negative.")

        for field_name in (
            "total_steps",
            "tool_calls_total",
            "tool_calls_successful",
            "tool_calls_failed",
            "invalid_tool_calls",
            "governance_denials",
            "rag_queries_total",
            "rag_sources_retrieved_total",
            "final_answer_length",
            "rag_sources_available_count",
            "rag_unique_chunks_count",
            "context_assembly_events_total",
            "context_messages_total",
            "context_estimated_tokens_total",
            "context_estimated_tokens_max",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


@dataclass(frozen=True)
class AgentContextQualityAssessment:
    """Deterministic context-quality assessment derived from run evidence."""

    budget_compliant: bool | None
    retrieval_evidence_present: bool | None
    has_semantic_memory_sources: bool | None
    has_episodic_memory_sources: bool | None
    has_working_memory_sources: bool | None
    has_chat_history_sources: bool | None
    has_tool_result_sources: bool | None
    context_source_profile_changes: int
    assemblies_total: int
    messages_total: int
    estimated_tokens_total: int
    estimated_tokens_max: int
    source_counts: dict[str, int]
    minimum_estimated_remaining_after_context: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "context_source_profile_changes",
            "assemblies_total",
            "messages_total",
            "estimated_tokens_total",
            "estimated_tokens_max",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")

    def as_dict(self) -> dict[str, object]:
        return {
            "budget_compliant": self.budget_compliant,
            "retrieval_evidence_present": self.retrieval_evidence_present,
            "has_semantic_memory_sources": self.has_semantic_memory_sources,
            "has_episodic_memory_sources": self.has_episodic_memory_sources,
            "has_working_memory_sources": self.has_working_memory_sources,
            "has_chat_history_sources": self.has_chat_history_sources,
            "has_tool_result_sources": self.has_tool_result_sources,
            "context_source_profile_changes": self.context_source_profile_changes,
            "assemblies_total": self.assemblies_total,
            "messages_total": self.messages_total,
            "estimated_tokens_total": self.estimated_tokens_total,
            "estimated_tokens_max": self.estimated_tokens_max,
            "minimum_estimated_remaining_after_context": (
                self.minimum_estimated_remaining_after_context
            ),
            "source_counts": dict(self.source_counts),
        }


@dataclass(frozen=True)
class AgentEvaluationMetrics:
    """Deterministic aggregate metrics derived from agent run evidence."""

    execution_time_ms: float
    steps_total: int
    tool_calls_total: int
    tool_calls_successful: int
    tool_calls_failed: int
    invalid_tool_calls: int
    governance_denials: int
    task_completed: bool
    rag_queries_total: int = 0
    rag_sources_retrieved_total: int = 0
    has_final_answer: bool = False
    final_answer_length: int = 0
    has_rag_provenance: bool = False
    has_rag_sources_available: bool = False
    rag_sources_available_count: int = 0
    rag_unique_chunks_count: int = 0
    retrieval_score_min: float | None = None
    retrieval_score_max: float | None = None
    retrieval_score_avg: float | None = None
    reranker_score_min: float | None = None
    reranker_score_max: float | None = None
    reranker_score_avg: float | None = None

    def __post_init__(self) -> None:
        if self.execution_time_ms < 0:
            raise ValueError("execution_time_ms must be non-negative.")

        for field_name in (
            "steps_total",
            "tool_calls_total",
            "tool_calls_successful",
            "tool_calls_failed",
            "invalid_tool_calls",
            "governance_denials",
            "rag_queries_total",
            "rag_sources_retrieved_total",
            "final_answer_length",
            "rag_sources_available_count",
            "rag_unique_chunks_count",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "execution_time_ms": self.execution_time_ms,
            "steps_total": self.steps_total,
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_successful": self.tool_calls_successful,
            "tool_calls_failed": self.tool_calls_failed,
            "invalid_tool_calls": self.invalid_tool_calls,
            "governance_denials": self.governance_denials,
            "rag_queries_total": self.rag_queries_total,
            "rag_sources_retrieved_total": self.rag_sources_retrieved_total,
            "has_final_answer": self.has_final_answer,
            "final_answer_length": self.final_answer_length,
            "has_rag_provenance": self.has_rag_provenance,
            "has_rag_sources_available": self.has_rag_sources_available,
            "rag_sources_available_count": self.rag_sources_available_count,
            "rag_unique_chunks_count": self.rag_unique_chunks_count,
            "retrieval_score_min": self.retrieval_score_min,
            "retrieval_score_max": self.retrieval_score_max,
            "retrieval_score_avg": self.retrieval_score_avg,
            "reranker_score_min": self.reranker_score_min,
            "reranker_score_max": self.reranker_score_max,
            "reranker_score_avg": self.reranker_score_avg,
            "task_completed": self.task_completed,
        }
