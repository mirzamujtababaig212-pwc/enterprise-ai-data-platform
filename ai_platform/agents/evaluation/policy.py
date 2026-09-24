from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_platform.agents.evaluation.models import AgentEvaluationMetrics


@dataclass(frozen=True)
class AgentEvaluationPolicy:
    """Configurable quality thresholds for agent execution evaluation."""

    max_execution_time_ms: float | None = None
    max_steps_per_run: int | None = None
    max_invalid_tool_calls: int | None = None
    allow_governance_denials: bool = False
    require_task_completed: bool = True
    name: str | None = None

    def __post_init__(self) -> None:
        if self.max_execution_time_ms is not None and self.max_execution_time_ms < 0:
            raise ValueError("max_execution_time_ms must be non-negative.")

        if self.max_steps_per_run is not None and self.max_steps_per_run < 0:
            raise ValueError("max_steps_per_run must be non-negative.")

        if self.max_invalid_tool_calls is not None and self.max_invalid_tool_calls < 0:
            raise ValueError("max_invalid_tool_calls must be non-negative.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_execution_time_ms": self.max_execution_time_ms,
            "max_steps_per_run": self.max_steps_per_run,
            "max_invalid_tool_calls": self.max_invalid_tool_calls,
            "allow_governance_denials": self.allow_governance_denials,
            "require_task_completed": self.require_task_completed,
            "name": self.name,
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

        return AgentQualityGateResult(
            passed=not violations,
            violations=tuple(violations),
        )
