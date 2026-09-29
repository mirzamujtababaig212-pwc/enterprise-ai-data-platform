from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_platform.agents.evaluation.answer_evaluation import AgentAnswerEvaluation
from ai_platform.agents.evaluation.models import AgentEvaluationMetrics


@dataclass(frozen=True)
class AgentEvaluationPolicy:
    """Configurable quality thresholds for agent execution evaluation."""

    max_execution_time_ms: float | None = None
    max_steps_per_run: int | None = None
    max_invalid_tool_calls: int | None = None
    min_retrieval_score: float | None = None
    min_reranker_score: float | None = None
    allow_governance_denials: bool = False
    require_task_completed: bool = True
    require_answer_match: bool = False
    name: str | None = None
    policy_id: str | None = None
    policy_version: str | None = None

    def __post_init__(self) -> None:
        if self.max_execution_time_ms is not None and self.max_execution_time_ms < 0:
            raise ValueError("max_execution_time_ms must be non-negative.")

        if self.max_steps_per_run is not None and self.max_steps_per_run < 0:
            raise ValueError("max_steps_per_run must be non-negative.")

        if self.max_invalid_tool_calls is not None and self.max_invalid_tool_calls < 0:
            raise ValueError("max_invalid_tool_calls must be non-negative.")

        if self.min_retrieval_score is not None and self.min_retrieval_score < 0:
            raise ValueError("min_retrieval_score must be non-negative.")

        if self.min_reranker_score is not None and self.min_reranker_score < 0:
            raise ValueError("min_reranker_score must be non-negative.")

        if self.policy_id is not None and not self.policy_id.strip():
            raise ValueError("policy_id must not be blank.")

        if self.policy_version is not None and not self.policy_version.strip():
            raise ValueError("policy_version must not be blank.")

        if (self.policy_id is None) != (self.policy_version is None):
            raise ValueError("policy_id and policy_version must be provided together.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_execution_time_ms": self.max_execution_time_ms,
            "max_steps_per_run": self.max_steps_per_run,
            "max_invalid_tool_calls": self.max_invalid_tool_calls,
            "min_retrieval_score": self.min_retrieval_score,
            "min_reranker_score": self.min_reranker_score,
            "allow_governance_denials": self.allow_governance_denials,
            "require_task_completed": self.require_task_completed,
            "require_answer_match": self.require_answer_match,
            "name": self.name,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class AgentQualityGateResult:
    """Evaluation result produced by applying an evaluation policy."""

    passed: bool
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": list(self.violations),
        }


class AgentQualityGateEvaluator:
    """Deterministically evaluates agent metrics against a policy."""

    @staticmethod
    def evaluate(
        metrics: AgentEvaluationMetrics,
        policy: AgentEvaluationPolicy,
        answer_evaluation: AgentAnswerEvaluation | None = None,
    ) -> AgentQualityGateResult:
        violations: list[str] = []

        if policy.require_task_completed and not metrics.task_completed:
            violations.append("Agent task execution did not complete successfully.")

        if (
            policy.max_execution_time_ms is not None
            and metrics.execution_time_ms > policy.max_execution_time_ms
        ):
            violations.append(
                f"Execution time ({metrics.execution_time_ms}ms) exceeded "
                f"maximum threshold ({policy.max_execution_time_ms}ms)."
            )

        if policy.max_steps_per_run is not None and metrics.steps_total > policy.max_steps_per_run:
            violations.append(
                f"Total steps ({metrics.steps_total}) exceeded "
                f"maximum threshold ({policy.max_steps_per_run})."
            )

        if (
            policy.max_invalid_tool_calls is not None
            and metrics.invalid_tool_calls > policy.max_invalid_tool_calls
        ):
            violations.append(
                f"Invalid tool calls ({metrics.invalid_tool_calls}) exceeded "
                f"allowed maximum ({policy.max_invalid_tool_calls})."
            )

        if not policy.allow_governance_denials and metrics.governance_denials > 0:
            violations.append(
                f"Execution triggered {metrics.governance_denials} " "governance denial(s)."
            )

        if policy.min_retrieval_score is not None:
            if metrics.retrieval_score_avg is None:
                violations.append(
                    "Average retrieval score was unavailable but minimum threshold "
                    f"({policy.min_retrieval_score:.2f}) is required."
                )
            elif metrics.retrieval_score_avg < policy.min_retrieval_score:
                violations.append(
                    f"Average retrieval score ({metrics.retrieval_score_avg:.2f}) was below "
                    f"minimum threshold ({policy.min_retrieval_score:.2f})."
                )

        if policy.min_reranker_score is not None:
            if metrics.reranker_score_avg is None:
                violations.append(
                    "Average reranker score was unavailable but minimum threshold "
                    f"({policy.min_reranker_score:.2f}) is required."
                )
            elif metrics.reranker_score_avg < policy.min_reranker_score:
                violations.append(
                    f"Average reranker score ({metrics.reranker_score_avg:.2f}) was below "
                    f"minimum threshold ({policy.min_reranker_score:.2f})."
                )

        if policy.require_answer_match:
            if answer_evaluation is None or not answer_evaluation.evaluated:
                violations.append(
                    "Answer match was required but answer evaluation was not performed."
                )
            elif not answer_evaluation.exact_match:
                violations.append("Final answer did not match the expected answer.")

        return AgentQualityGateResult(
            passed=not violations,
            violations=tuple(violations),
        )
