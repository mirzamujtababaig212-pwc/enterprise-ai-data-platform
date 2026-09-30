from __future__ import annotations

from dataclasses import dataclass


ANSWER_NORMALIZATION = "whitespace_casefold"


def _normalize_answer(value: str) -> str:
    """Normalize answer text for deterministic exact-match evaluation."""
    return " ".join(value.strip().split()).casefold()


@dataclass(frozen=True)
class AgentAnswerEvaluation:
    """Answer evaluation result containing deterministic and optional semantic signals."""

    evaluated: bool
    exact_match: bool | None
    normalization: str = ANSWER_NORMALIZATION
    semantic_evaluated: bool = False
    semantic_score: float | None = None
    semantic_passed: bool | None = None
    semantic_method: str | None = None
    evaluator_model: str | None = None
    evaluator_provider: str | None = None

    def __post_init__(self) -> None:
        if not self.evaluated and self.exact_match is not None:
            raise ValueError("exact_match must be None when answer evaluation is not evaluated.")

        if self.evaluated and self.exact_match is None:
            raise ValueError("exact_match must be set when answer evaluation is evaluated.")

        if not self.semantic_evaluated:
            if self.semantic_score is not None:
                raise ValueError(
                    "semantic_score must be None when semantic evaluation is not evaluated."
                )

            if self.semantic_passed is not None:
                raise ValueError(
                    "semantic_passed must be None when semantic evaluation is not evaluated."
                )

            if self.semantic_method is not None:
                raise ValueError(
                    "semantic_method must be None when semantic evaluation is not evaluated."
                )

            if self.evaluator_model is not None:
                raise ValueError(
                    "evaluator_model must be None when semantic evaluation is not evaluated."
                )

            if self.evaluator_provider is not None:
                raise ValueError(
                    "evaluator_provider must be None when semantic evaluation is not evaluated."
                )

        else:
            if self.semantic_score is None:
                raise ValueError(
                    "semantic_score must be set when semantic evaluation is evaluated."
                )

            if not 0.0 <= self.semantic_score <= 1.0:
                raise ValueError("semantic_score must be between 0.0 and 1.0.")

            if self.semantic_passed is None:
                raise ValueError(
                    "semantic_passed must be set when semantic evaluation is evaluated."
                )

            if self.semantic_method is None or not self.semantic_method.strip():
                raise ValueError(
                    "semantic_method must be set when semantic evaluation is evaluated."
                )

    def as_dict(self) -> dict[str, bool | float | str | None]:
        return {
            "evaluated": self.evaluated,
            "exact_match": self.exact_match,
            "normalization": self.normalization,
            "semantic_evaluated": self.semantic_evaluated,
            "semantic_score": self.semantic_score,
            "semantic_passed": self.semantic_passed,
            "semantic_method": self.semantic_method,
            "evaluator_model": self.evaluator_model,
            "evaluator_provider": self.evaluator_provider,
        }


class AgentAnswerEvaluator:
    """Evaluate an extracted agent answer against an expected answer."""

    @staticmethod
    def evaluate(
        *,
        actual_answer: str | None,
        expected_answer: str | None,
    ) -> AgentAnswerEvaluation:
        if expected_answer is None:
            return AgentAnswerEvaluation(
                evaluated=False,
                exact_match=None,
            )

        normalized_expected = _normalize_answer(expected_answer)

        if actual_answer is None:
            return AgentAnswerEvaluation(
                evaluated=True,
                exact_match=False,
            )

        normalized_actual = _normalize_answer(actual_answer)

        return AgentAnswerEvaluation(
            evaluated=True,
            exact_match=normalized_actual == normalized_expected,
        )
