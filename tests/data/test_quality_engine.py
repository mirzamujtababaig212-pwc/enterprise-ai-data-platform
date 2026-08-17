"""Tests for data quality orchestration."""

import pytest

from ai_platform.data.quality import (
    AllowedValuesRule,
    NotNullRule,
    NumericRangeRule,
    QualityEngine,
    UniqueRule,
)


def test_quality_engine_with_all_passing_rules() -> None:
    engine = QualityEngine(
        [
            NotNullRule("customer_id"),
            UniqueRule("customer_id"),
            AllowedValuesRule(
                "status",
                {"ACTIVE", "INACTIVE"},
            ),
            NumericRangeRule(
                "score",
                minimum=0,
                maximum=100,
            ),
        ]
    )

    result = engine.evaluate(
        [
            {
                "customer_id": 1,
                "status": "ACTIVE",
                "score": 90,
            },
            {
                "customer_id": 2,
                "status": "INACTIVE",
                "score": 75,
            },
        ]
    )

    assert result["overall_passed"] is True
    assert result["total_rules"] == 4
    assert result["passed_rules"] == 4
    assert result["failed_rules"] == 0
    assert len(result["results"]) == 4


def test_quality_engine_detects_failed_rules() -> None:
    engine = QualityEngine(
        [
            NotNullRule("customer_id"),
            UniqueRule("customer_id"),
            AllowedValuesRule(
                "status",
                {"ACTIVE", "INACTIVE"},
            ),
        ]
    )

    result = engine.evaluate(
        [
            {
                "customer_id": 1,
                "status": "ACTIVE",
            },
            {
                "customer_id": 1,
                "status": "DELETED",
            },
            {
                "customer_id": None,
                "status": "ACTIVE",
            },
        ]
    )

    assert result["overall_passed"] is False
    assert result["total_rules"] == 3
    assert result["passed_rules"] == 0
    assert result["failed_rules"] == 3


def test_quality_engine_with_mixed_results() -> None:
    engine = QualityEngine(
        [
            NotNullRule("customer_id"),
            NumericRangeRule(
                "score",
                minimum=0,
                maximum=100,
            ),
        ]
    )

    result = engine.evaluate(
        [
            {
                "customer_id": 1,
                "score": 50,
            },
            {
                "customer_id": 2,
                "score": 150,
            },
        ]
    )

    assert result["overall_passed"] is False
    assert result["total_rules"] == 2
    assert result["passed_rules"] == 1
    assert result["failed_rules"] == 1


def test_quality_engine_supports_empty_rules() -> None:
    engine = QualityEngine([])

    result = engine.evaluate(
        [
            {"id": 1},
            {"id": 2},
        ]
    )

    assert result["overall_passed"] is True
    assert result["total_rules"] == 0
    assert result["passed_rules"] == 0
    assert result["failed_rules"] == 0
    assert result["results"] == []


def test_quality_engine_rejects_invalid_rule() -> None:
    with pytest.raises(
        TypeError,
        match="DataQualityRule protocol",
    ):
        QualityEngine([object()])


def test_quality_engine_report_is_serializable() -> None:
    engine = QualityEngine(
        [
            NotNullRule("customer_id"),
        ]
    )

    result = engine.evaluate(
        [
            {"customer_id": 1},
        ]
    )

    assert isinstance(result, dict)
    assert isinstance(result["results"], list)
    assert result["results"][0]["rule_name"] == ("not_null:customer_id")
