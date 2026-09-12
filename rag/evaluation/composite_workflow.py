from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from rag.evaluation.comparison.regression_policy import RetrievalRegressionPolicy
from rag.evaluation.composite_release import (
    CompositeEvaluationReleaseDecision,
    CompositeEvaluationReleaseGate,
)
from rag.evaluation.external.models import ExternalEvaluationResult
from rag.evaluation.external.release import (
    ExternalEvaluationReleaseGate,
    ExternalEvaluationReleasePolicy,
)
from rag.evaluation.external.policy import ExternalEvaluationPolicy
from rag.evaluation.release import RetrievalEvaluationReleaseGate
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.workflow import RetrievalEvaluationWorkflowResult


@dataclass(frozen=True)
class CompositeEvaluationWorkflowResult:
    """Complete evaluation-run artifact and its composite release decision."""

    run: RetrievalEvaluationRun
    release_decision: CompositeEvaluationReleaseDecision

    def as_dict(self) -> dict[str, object]:
        return {
            "run": self.run.as_dict(),
            "release_decision": self.release_decision.as_dict(),
        }


class CompositeEvaluationWorkflow:
    """
    Assemble native and external evaluation evidence into one release decision.

    Native evaluation and regression remain owned by the existing retrieval
    evaluation workflow. External evidence is attached separately and evaluated
    against its own metric policy. The final release decision is then composed
    without changing the native release semantics.
    """

    @staticmethod
    def build_run(
        *,
        result: RetrievalEvaluationWorkflowResult,
        run_id: str,
        created_at: datetime,
        baseline=None,
        baseline_run_id: str | None = None,
        baselines: Iterable[RetrievalEvaluationRun] | None = None,
        regression_policy: RetrievalRegressionPolicy | None = None,
        external_evaluations: Iterable[ExternalEvaluationResult] = (),
        external_policy: ExternalEvaluationPolicy | None = None,
    ) -> RetrievalEvaluationRun:
        """Build the immutable evaluation run from all available evidence."""
        run = result.to_run(
            run_id=run_id,
            created_at=created_at,
            baseline=baseline,
            baseline_run_id=baseline_run_id,
            baselines=baselines,
            regression_policy=regression_policy,
        )

        for external_evaluation in external_evaluations:
            run = run.with_external_evaluation(external_evaluation)

        if external_policy is not None:
            run = run.with_external_quality_gate(external_policy)

        return run

    @staticmethod
    def release_decision(
        *,
        run: RetrievalEvaluationRun,
        external_release_policy: ExternalEvaluationReleasePolicy,
    ) -> CompositeEvaluationReleaseDecision:
        """Evaluate native and external release decisions together."""
        native = RetrievalEvaluationReleaseGate.evaluate(run)

        external = ExternalEvaluationReleaseGate.evaluate(
            quality_gate=run.external_quality_gate,
            policy=external_release_policy,
        )

        return CompositeEvaluationReleaseGate.evaluate(
            run=run,
            native=native,
            external=external,
        )

    @classmethod
    def run(
        cls,
        *,
        result: RetrievalEvaluationWorkflowResult,
        run_id: str,
        created_at: datetime,
        external_release_policy: ExternalEvaluationReleasePolicy,
        baseline=None,
        baseline_run_id: str | None = None,
        baselines: Iterable[RetrievalEvaluationRun] | None = None,
        regression_policy: RetrievalRegressionPolicy | None = None,
        external_evaluations: Iterable[ExternalEvaluationResult] = (),
        external_policy: ExternalEvaluationPolicy | None = None,
    ) -> CompositeEvaluationWorkflowResult:
        """Build a complete evaluation run and its final release decision."""
        run = cls.build_run(
            result=result,
            run_id=run_id,
            created_at=created_at,
            baseline=baseline,
            baseline_run_id=baseline_run_id,
            baselines=baselines,
            regression_policy=regression_policy,
            external_evaluations=external_evaluations,
            external_policy=external_policy,
        )

        release_decision = cls.release_decision(
            run=run,
            external_release_policy=external_release_policy,
        )

        return CompositeEvaluationWorkflowResult(
            run=run,
            release_decision=release_decision,
        )
