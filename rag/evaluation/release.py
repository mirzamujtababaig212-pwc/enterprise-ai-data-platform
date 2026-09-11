from __future__ import annotations

from dataclasses import dataclass

from rag.evaluation.run import RetrievalEvaluationRun


@dataclass(frozen=True)
class RetrievalEvaluationReleaseDecision:
    """Immutable release decision derived from an evaluation run."""

    run_id: str
    passed: bool
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "errors": self.errors,
        }


class RetrievalEvaluationReleaseGate:
    """Enforce the final release decision for a retrieval evaluation run."""

    @staticmethod
    def evaluate(
        run: RetrievalEvaluationRun,
    ) -> RetrievalEvaluationReleaseDecision:
        errors: list[str] = []

        if not run.passed:
            errors.append("retrieval quality gate failed")

        if run.regression is not None and not run.regression.passed:
            errors.append("retrieval regression policy failed")

        return RetrievalEvaluationReleaseDecision(
            run_id=run.run_id,
            passed=not errors,
            errors=tuple(errors),
        )
