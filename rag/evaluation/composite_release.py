from __future__ import annotations

from dataclasses import dataclass

from rag.evaluation.external.release import ExternalEvaluationReleaseDecision
from rag.evaluation.release import RetrievalEvaluationReleaseDecision
from rag.evaluation.run import RetrievalEvaluationRun


@dataclass(frozen=True)
class CompositeEvaluationReleaseDecision:
    """
    Final release decision combining native retrieval evaluation and
    external evaluation evidence.
    """

    run_id: str
    native: RetrievalEvaluationReleaseDecision
    external: ExternalEvaluationReleaseDecision
    passed: bool
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "errors": self.errors,
            "native": self.native.as_dict(),
            "external": self.external.as_dict(),
        }


class CompositeEvaluationReleaseGate:
    """
    Combine native and external evaluation release decisions.

    The existing native release decision remains authoritative for
    native retrieval quality and regression. External evaluation is
    evaluated independently and then composed here.
    """

    @staticmethod
    def evaluate(
        *,
        run: RetrievalEvaluationRun,
        native: RetrievalEvaluationReleaseDecision,
        external: ExternalEvaluationReleaseDecision,
    ) -> CompositeEvaluationReleaseDecision:
        if native.run_id != run.run_id:
            raise ValueError("native release decision run_id must match evaluation run")

        errors: list[str] = []

        if not native.passed:
            errors.extend(f"native: {error}" for error in native.errors)

        if not external.passed:
            errors.extend(f"external: {error}" for error in external.errors)

        return CompositeEvaluationReleaseDecision(
            run_id=run.run_id,
            native=native,
            external=external,
            passed=not errors,
            errors=tuple(errors),
        )
