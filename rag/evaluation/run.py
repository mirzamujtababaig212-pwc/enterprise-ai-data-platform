from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.evaluation.comparison.regression_policy import (
        RetrievalRegressionPolicy,
        RetrievalRegressionPolicyResult,
    )
    from rag.evaluation.comparison.run_comparator import (
        RetrievalEvaluationRunComparison,
    )
from rag.evaluation.external.models import ExternalEvaluationResult
from rag.evaluation.lineage import RetrievalEvaluationLineage
from rag.evaluation.models import RetrievalEvaluationResult
from rag.evaluation.quality_gate import RetrievalQualityGateResult


@dataclass(frozen=True)
class RetrievalEvaluationRegression:
    """
    Regression evaluation attached to a candidate evaluation run.

    Contains the baseline identity, metric comparison, applied tolerance
    policy, and resulting regression decision.
    """

    baseline_run_id: str
    candidate_run_id: str
    comparison: RetrievalEvaluationRunComparison
    policy: RetrievalRegressionPolicy
    result: RetrievalRegressionPolicyResult

    @property
    def passed(self) -> bool:
        return self.result.passed

    def as_dict(self) -> dict[str, object]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "passed": self.passed,
            "comparison": self.comparison.as_dict(),
            "policy": self.policy.as_dict(),
            "result": self.result.as_dict(),
        }


@dataclass(frozen=True)
class RetrievalEvaluationRun:
    """
    Immutable artifact representing one retrieval evaluation run.

    The run contains aggregate evaluation outputs and provenance, but does not
    contain raw query text, retrieved document content, embeddings, or per-sample
    evaluation payloads.
    """

    run_id: str
    created_at: datetime
    lineage: RetrievalEvaluationLineage
    evaluation: RetrievalEvaluationResult
    quality_gate: RetrievalQualityGateResult
    regression: RetrievalEvaluationRegression | None = None
    external_evaluations: tuple[ExternalEvaluationResult, ...] = ()

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")

        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone-aware")

        if self.regression is not None and self.regression.candidate_run_id != self.run_id:
            raise ValueError("regression comparison candidate_run_id must match run_id")

    @property
    def passed(self) -> bool:
        """Return the quality-gate decision."""
        return self.quality_gate.passed

    @property
    def release_passed(self) -> bool:
        """
        Return the final release decision.

        A run must pass its quality gate. If regression evaluation is present,
        it must also pass the configured regression policy.
        """
        return self.passed and (self.regression is None or self.regression.passed)

    def with_regression(
        self,
        *,
        baseline: RetrievalEvaluationRun,
        policy: RetrievalRegressionPolicy,
    ) -> RetrievalEvaluationRun:
        """
        Return a new immutable run with regression evaluation attached.

        The original run remains unchanged.
        """
        from rag.evaluation.comparison import (
            RetrievalEvaluationRunComparator,
            RetrievalRegressionPolicyEvaluator,
        )

        comparison = RetrievalEvaluationRunComparator.compare(
            baseline,
            self,
        )

        result = RetrievalRegressionPolicyEvaluator.evaluate(
            comparison,
            policy,
        )

        regression = RetrievalEvaluationRegression(
            baseline_run_id=baseline.run_id,
            candidate_run_id=self.run_id,
            comparison=comparison,
            policy=policy,
            result=result,
        )

        return replace(
            self,
            regression=regression,
        )

    def with_external_evaluation(
        self,
        result: ExternalEvaluationResult,
    ) -> RetrievalEvaluationRun:
        """
        Return a new immutable run with external evaluation evidence attached.

        The original run remains unchanged. External evaluation evidence does
        not affect the native quality-gate or release decisions.
        """
        return replace(
            self,
            external_evaluations=(
                *self.external_evaluations,
                result,
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "passed": self.passed,
            "release_passed": self.release_passed,
            "lineage": self.lineage.as_dict(),
            "evaluation": self.evaluation.as_dict(),
            "quality_gate": self.quality_gate.as_dict(),
            "regression": (self.regression.as_dict() if self.regression is not None else None),
            "external_evaluations": [result.as_dict() for result in self.external_evaluations],
        }
