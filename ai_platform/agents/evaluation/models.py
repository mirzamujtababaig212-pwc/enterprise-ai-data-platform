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
    rag_sources_available_count: int = 0
    rag_unique_chunks_count: int = 0
    effective_model: str | None = None
    effective_provider: str | None = None
    model_policy_id: str | None = None
    model_policy_version: str | None = None
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
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative.")


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
    rag_sources_available_count: int = 0
    rag_unique_chunks_count: int = 0

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
            "rag_sources_available_count": self.rag_sources_available_count,
            "rag_unique_chunks_count": self.rag_unique_chunks_count,
            "task_completed": self.task_completed,
        }
