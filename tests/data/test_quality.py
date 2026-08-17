"""Tests for reusable data quality rules."""

from ai_platform.data.contracts import DataQualityRule
from ai_platform.data.quality import (
    AllowedValuesRule,
    NotNullRule,
    NumericRangeRule,
    UniqueRule,
)


def test_not_null_rule_implements_contract() -> None:
    rule = NotNullRule("customer_id")

    assert isinstance(rule, DataQualityRule)


def test_not_null_rule_passes_valid_records() -> None:
    rule = NotNullRule("customer_id")

    result = rule.evaluate(
        [
            {"customer_id": 1},
            {"customer_id": 2},
        ]
    )

    assert result["passed"] is True
    assert result["checked_records"] == 2
    assert result["failed_records"] == 0


def test_not_null_rule_detects_null() -> None:
    rule = NotNullRule("customer_id")

    result = rule.evaluate(
        [
            {"customer_id": 1},
            {"customer_id": None},
            {},
        ]
    )

    assert result["passed"] is False
    assert result["checked_records"] == 3
    assert result["failed_records"] == 2


def test_unique_rule_passes_unique_values() -> None:
    rule = UniqueRule("customer_id")

    result = rule.evaluate(
        [
            {"customer_id": 1},
            {"customer_id": 2},
            {"customer_id": 3},
        ]
    )

    assert result["passed"] is True
    assert result["failed_records"] == 0


def test_unique_rule_detects_duplicates() -> None:
    rule = UniqueRule("customer_id")

    result = rule.evaluate(
        [
            {"customer_id": 1},
            {"customer_id": 1},
            {"customer_id": 2},
        ]
    )

    assert result["passed"] is False
    assert result["failed_records"] == 1


def test_allowed_values_rule() -> None:
    rule = AllowedValuesRule(
        "status",
        {"ACTIVE", "INACTIVE"},
    )

    result = rule.evaluate(
        [
            {"status": "ACTIVE"},
            {"status": "INACTIVE"},
        ]
    )

    assert result["passed"] is True


def test_allowed_values_rule_rejects_invalid_value() -> None:
    rule = AllowedValuesRule(
        "status",
        {"ACTIVE", "INACTIVE"},
    )

    result = rule.evaluate(
        [
            {"status": "ACTIVE"},
            {"status": "DELETED"},
        ]
    )

    assert result["passed"] is False
    assert result["failed_records"] == 1


def test_numeric_range_rule() -> None:
    rule = NumericRangeRule(
        "score",
        minimum=0,
        maximum=100,
    )

    result = rule.evaluate(
        [
            {"score": 0},
            {"score": 50},
            {"score": 100},
        ]
    )

    assert result["passed"] is True


def test_numeric_range_rule_rejects_out_of_range_values() -> None:
    rule = NumericRangeRule(
        "score",
        minimum=0,
        maximum=100,
    )

    result = rule.evaluate(
        [
            {"score": 50},
            {"score": -1},
            {"score": 101},
        ]
    )

    assert result["passed"] is False
    assert result["failed_records"] == 2


def test_numeric_range_rule_rejects_non_numeric_values() -> None:
    rule = NumericRangeRule(
        "score",
        minimum=0,
        maximum=100,
    )

    result = rule.evaluate(
        [
            {"score": "not-a-number"},
        ]
    )

    assert result["passed"] is False
    assert result["failed_records"] == 1


def test_quality_rule_metadata() -> None:
    rule = NotNullRule("customer_id")

    result = rule.evaluate(
        [
            {"customer_id": 1},
        ]
    )

    assert result["rule_name"] == "not_null:customer_id"
    assert result["checked_records"] == 1
    assert result["failed_records"] == 0
    assert "passed" in result
    assert "message" in result


def test_invalid_numeric_range_configuration() -> None:
    try:
        NumericRangeRule(
            "score",
            minimum=100,
            maximum=0,
        )
    except ValueError as exc:
        assert "minimum cannot exceed maximum" in str(exc)
    else:
        raise AssertionError("Expected invalid numeric range configuration to fail.")
