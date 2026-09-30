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
        "semantic_evaluated": False,
        "semantic_score": None,
        "semantic_passed": None,
        "semantic_method": None,
        "evaluator_model": None,
        "evaluator_provider": None,
    }

    assert "Sensitive answer content" not in str(result.as_dict())


def test_semantic_evaluation_can_be_recorded_observationally() -> None:
    result = AgentAnswerEvaluation(
        evaluated=True,
        exact_match=False,
        semantic_evaluated=True,
        semantic_score=0.92,
        semantic_passed=True,
        semantic_method="llm_judge_v1",
        evaluator_model="gpt-4.1-mini",
        evaluator_provider="openai",
    )

    assert result.semantic_evaluated is True
    assert result.semantic_score == 0.92
    assert result.semantic_passed is True
    assert result.semantic_method == "llm_judge_v1"
    assert result.evaluator_model == "gpt-4.1-mini"
    assert result.evaluator_provider == "openai"


def test_answer_evaluation_defaults_to_no_semantic_evaluation() -> None:
    result = AgentAnswerEvaluator.evaluate(
        actual_answer="Expected response",
        expected_answer="Expected response",
    )

    assert result.semantic_evaluated is False
    assert result.semantic_score is None
    assert result.semantic_passed is None
    assert result.semantic_method is None
    assert result.evaluator_model is None
    assert result.evaluator_provider is None


def test_semantic_evaluation_serializes_without_answer_text() -> None:
    result = AgentAnswerEvaluation(
        evaluated=True,
        exact_match=False,
        semantic_evaluated=True,
        semantic_score=0.87,
        semantic_passed=True,
        semantic_method="llm_judge_v1",
        evaluator_model="gpt-4.1-mini",
        evaluator_provider="openai",
    )

    assert result.as_dict() == {
        "evaluated": True,
        "exact_match": False,
        "normalization": ANSWER_NORMALIZATION,
        "semantic_evaluated": True,
        "semantic_score": 0.87,
        "semantic_passed": True,
        "semantic_method": "llm_judge_v1",
        "evaluator_model": "gpt-4.1-mini",
        "evaluator_provider": "openai",
    }


def test_semantic_evaluation_rejects_score_outside_unit_interval() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_score must be between 0.0 and 1.0",
    ):
        AgentAnswerEvaluation(
            evaluated=True,
            exact_match=False,
            semantic_evaluated=True,
            semantic_score=1.01,
            semantic_passed=True,
            semantic_method="llm_judge_v1",
        )


def test_semantic_evaluation_rejects_incomplete_state() -> None:
    with pytest.raises(
        ValueError,
        match="semantic_score must be set",
    ):
        AgentAnswerEvaluation(
            evaluated=True,
            exact_match=False,
            semantic_evaluated=True,
            semantic_passed=True,
            semantic_method="llm_judge_v1",
        )
