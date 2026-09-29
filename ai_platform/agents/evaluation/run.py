from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ai_platform.agents.evaluation.answer_evaluation import AgentAnswerEvaluation
from ai_platform.agents.evaluation.models import (
    AgentContextQualityAssessment,
    AgentEvaluationMetrics,
)
from ai_platform.agents.evaluation.policy import (
    AgentEvaluationPolicy,
    AgentQualityGateResult,
)


@dataclass(frozen=True)
class AgentEvaluationLineage:
    """Provenance for the agent execution being evaluated."""

    evaluated_run_id: str
    agent_name: str
    agent_version: str | None
    tenant_id: str | None
    effective_model: str | None = None
    effective_provider: str | None = None
    model_policy_id: str | None = None
    model_policy_version: str | None = None

    def __post_init__(self) -> None:
        if not self.evaluated_run_id.strip():
            raise ValueError("evaluated_run_id must not be empty.")

        if not self.agent_name.strip():
            raise ValueError("agent_name must not be empty.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluated_run_id": self.evaluated_run_id,
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "tenant_id": self.tenant_id,
            "effective_model": self.effective_model,
            "effective_provider": self.effective_provider,
            "model_policy_id": self.model_policy_id,
            "model_policy_version": self.model_policy_version,
        }


@dataclass(frozen=True)
class AgentEvaluationRun:
    """Immutable evaluation artifact for one agent execution."""

    evaluation_run_id: str
    created_at: datetime
    lineage: AgentEvaluationLineage
    metrics: AgentEvaluationMetrics
    policy: AgentEvaluationPolicy
    quality_gate: AgentQualityGateResult
    answer_evaluation: AgentAnswerEvaluation | None = None
    context_quality: AgentContextQualityAssessment | None = None

    def __post_init__(self) -> None:
        if not self.evaluation_run_id.strip():
            raise ValueError("evaluation_run_id must not be empty.")

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware.")

    @property
    def passed(self) -> bool:
        return self.quality_gate.passed

    def as_dict(self) -> dict[str, Any]:
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "created_at": self.created_at.isoformat(),
            "passed": self.passed,
            "lineage": self.lineage.as_dict(),
            "metrics": self.metrics.as_dict(),
            "policy": self.policy.as_dict(),
            "quality_gate": self.quality_gate.as_dict(),
            "answer_evaluation": (
                self.answer_evaluation.as_dict() if self.answer_evaluation is not None else None
            ),
            "context_quality": (
                self.context_quality.as_dict() if self.context_quality is not None else None
            ),
        }
