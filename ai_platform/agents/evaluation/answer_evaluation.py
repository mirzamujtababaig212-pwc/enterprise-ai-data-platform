from __future__ import annotations

from dataclasses import dataclass


ANSWER_NORMALIZATION = "whitespace_casefold"


def _normalize_answer(value: str) -> str:
    """Normalize answer text for deterministic exact-match evaluation."""
    return " ".join(value.strip().split()).casefold()


@dataclass(frozen=True)
class AgentAnswerEvaluation:
    """Deterministic expected-answer evaluation result."""

    evaluated: bool
    exact_match: bool | None
    normalization: str = ANSWER_NORMALIZATION

    def __post_init__(self) -> None:
        if not self.evaluated and self.exact_match is not None:
            raise ValueError("exact_match must be None when answer evaluation is not evaluated.")

        if self.evaluated and self.exact_match is None:
            raise ValueError("exact_match must be set when answer evaluation is evaluated.")

    def as_dict(self) -> dict[str, bool | str | None]:
        return {
            "evaluated": self.evaluated,
            "exact_match": self.exact_match,
            "normalization": self.normalization,
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
