import pytest

from ai_platform.agents.evaluation.answer_evaluation import (
    ANSWER_NORMALIZATION,
    AgentAnswerEvaluation,
    AgentAnswerEvaluator,
)


def test_exact_match_ignores_case_and_repeated_whitespace() -> None:
    result = AgentAnswerEvaluator.evaluate(
        actual_answer="  The   vehicle exceeded the speed limit. ",
        expected_answer="the vehicle exceeded the speed limit.",
    )

    assert result == AgentAnswerEvaluation(
        evaluated=True,
        exact_match=True,
        normalization=ANSWER_NORMALIZATION,
    )


def test_exact_match_rejects_different_wording() -> None:
    result = AgentAnswerEvaluator.evaluate(
        actual_answer="The vehicle was above the speed threshold.",
        expected_answer="The vehicle exceeded the speed limit.",
    )

    assert result.evaluated is True
    assert result.exact_match is False


def test_missing_actual_answer_fails_evaluated_comparison() -> None:
    result = AgentAnswerEvaluator.evaluate(
        actual_answer=None,
        expected_answer="The vehicle exceeded the speed limit.",
    )

    assert result.evaluated is True
    assert result.exact_match is False


def test_missing_expected_answer_skips_evaluation() -> None:
    result = AgentAnswerEvaluator.evaluate(
        actual_answer="The vehicle exceeded the speed limit.",
        expected_answer=None,
    )

    assert result.evaluated is False
    assert result.exact_match is None


def test_blank_expected_answer_is_evaluated() -> None:
    result = AgentAnswerEvaluator.evaluate(
        actual_answer="",
        expected_answer="   ",
    )

    assert result.evaluated is True
    assert result.exact_match is True


def test_answer_evaluation_rejects_inconsistent_state() -> None:
    with pytest.raises(
        ValueError,
        match="exact_match must be None when answer evaluation is not evaluated",
    ):
        AgentAnswerEvaluation(
            evaluated=False,
            exact_match=True,
        )

    with pytest.raises(
        ValueError,
        match="exact_match must be set when answer evaluation is evaluated",
    ):
        AgentAnswerEvaluation(
            evaluated=True,
            exact_match=None,
        )


def test_answer_evaluation_serializes_without_answer_text() -> None:
    result = AgentAnswerEvaluator.evaluate(
        actual_answer="Sensitive answer content",
        expected_answer="Sensitive answer content",
    )

    assert result.as_dict() == {
        "evaluated": True,
        "exact_match": True,
        "normalization": ANSWER_NORMALIZATION,
    }

    assert "Sensitive answer content" not in str(result.as_dict())
