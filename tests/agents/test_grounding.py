from ai_platform.agents.evaluation.grounding import (
    GROUNDING_METHOD,
    AgentGroundingEvaluator,
)


def test_grounding_supports_answer_with_textual_source_overlap() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text="The vehicle battery temperature reached 42 degrees Celsius.",
        source_texts=[
            "Vehicle V001 telemetry shows the battery temperature reached 42 degrees Celsius."
        ],
    )

    assert result.evaluated is True
    assert result.supported is True
    assert result.support_ratio == 1.0
    assert result.supported_sources_total == 1
    assert result.source_candidates_total == 1
    assert result.method == GROUNDING_METHOD


def test_grounding_detects_partially_supported_answer() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text=(
            "The vehicle battery temperature reached 42 degrees Celsius. "
            "The vehicle was driven for 900 kilometers."
        ),
        source_texts=[
            "Vehicle V001 telemetry shows the battery temperature reached 42 degrees Celsius."
        ],
    )

    assert result.evaluated is True
    assert result.supported is False
    assert result.support_ratio == 0.5
    assert result.supported_sources_total == 1
    assert result.source_candidates_total == 1


def test_grounding_rejects_unsupported_answer() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text="The vehicle battery temperature reached 99 degrees Celsius.",
        source_texts=[
            "Vehicle V001 telemetry shows the battery temperature reached 42 degrees Celsius."
        ],
    )

    assert result.evaluated is True
    assert result.supported is False
    assert result.support_ratio == 0.0
    assert result.supported_sources_total == 0
    assert result.source_candidates_total == 1


def test_grounding_is_not_evaluated_without_answer() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text=None,
        source_texts=["Vehicle telemetry shows a battery temperature of 42 degrees Celsius."],
    )

    assert result.evaluated is False
    assert result.supported is None
    assert result.support_ratio is None
    assert result.supported_sources_total == 0
    assert result.source_candidates_total == 1


def test_grounding_is_not_evaluated_without_sources() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text="The vehicle battery temperature reached 42 degrees Celsius.",
        source_texts=[],
    )

    assert result.evaluated is False
    assert result.supported is None
    assert result.support_ratio is None
    assert result.supported_sources_total == 0
    assert result.source_candidates_total == 0


def test_grounding_serializes_only_aggregate_diagnostics() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text="The battery temperature reached 42 degrees Celsius.",
        source_texts=["Battery temperature reached 42 degrees Celsius."],
    )

    assert result.as_dict() == {
        "evaluated": True,
        "supported": True,
        "support_ratio": 1.0,
        "supported_sources_total": 1,
        "source_candidates_total": 1,
        "method": GROUNDING_METHOD,
    }


def test_grounding_handles_bulleted_answer_as_separate_claims() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text=(
            "- The battery temperature reached 42 degrees Celsius\n"
            "- The vehicle travelled for 900 kilometers"
        ),
        source_texts=[
            "Vehicle V001 telemetry shows the battery temperature reached 42 degrees Celsius."
        ],
    )

    assert result.evaluated is True
    assert result.supported is False
    assert result.support_ratio == 0.5
    assert result.supported_sources_total == 1
    assert result.source_candidates_total == 1


def test_grounding_does_not_reject_supported_number_when_source_has_other_numbers() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text="The battery temperature reached 42 degrees Celsius.",
        source_texts=[
            (
                "At 10:00 the battery temperature was 35 degrees Celsius. "
                "Later, the battery temperature reached 42 degrees Celsius."
            )
        ],
    )

    assert result.evaluated is True
    assert result.supported is True
    assert result.support_ratio == 1.0
    assert result.supported_sources_total == 1


def test_grounding_rejects_numeric_claim_when_supporting_source_statement_disagrees() -> None:
    result = AgentGroundingEvaluator.evaluate(
        answer_text="The battery temperature reached 99 degrees Celsius.",
        source_texts=[
            (
                "At 10:00 the battery temperature was 35 degrees Celsius. "
                "Later, the battery temperature reached 42 degrees Celsius."
            )
        ],
    )

    assert result.evaluated is True
    assert result.supported is False
    assert result.support_ratio == 0.0
    assert result.supported_sources_total == 0
