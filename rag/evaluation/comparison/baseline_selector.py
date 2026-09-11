from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from rag.evaluation.comparison.run_comparator import (
    RetrievalEvaluationRunComparator,
)
from rag.evaluation.run import RetrievalEvaluationRun


class RetrievalEvaluationBaselineSelectionError(ValueError):
    """Raised when an evaluation baseline cannot be selected."""


@dataclass(frozen=True)
class RetrievalEvaluationBaselineSelector:
    """Select an explicitly approved and compatible baseline evaluation run."""

    @staticmethod
    def select(
        *,
        candidate: RetrievalEvaluationRun,
        baselines: Iterable[RetrievalEvaluationRun],
        baseline_run_id: str,
    ) -> RetrievalEvaluationRun:
        if not baseline_run_id.strip():
            raise RetrievalEvaluationBaselineSelectionError("baseline_run_id must not be empty")

        if baseline_run_id == candidate.run_id:
            raise RetrievalEvaluationBaselineSelectionError(
                "candidate run cannot be used as its own baseline"
            )

        matches = [run for run in baselines if run.run_id == baseline_run_id]

        if not matches:
            raise RetrievalEvaluationBaselineSelectionError(
                f"baseline run not found: {baseline_run_id!r}"
            )

        if len(matches) > 1:
            raise RetrievalEvaluationBaselineSelectionError(
                f"multiple baseline runs found: {baseline_run_id!r}"
            )

        baseline = matches[0]

        try:
            RetrievalEvaluationRunComparator.validate_compatibility(
                baseline,
                candidate,
            )
        except ValueError as exc:
            raise RetrievalEvaluationBaselineSelectionError(
                f"baseline run is incompatible: {exc}"
            ) from exc

        return baseline
