from __future__ import annotations

import pandas as pd

from ml.platform import (
    FeatureContract,
    FeatureDefinition,
    FeatureValidator,
)


def _contract() -> FeatureContract:
    return FeatureContract(
        name="loan-default",
        features=(
            FeatureDefinition(
                name="income",
                dtype="float",
                min_value=0,
            ),
            FeatureDefinition(
                name="age",
                dtype="int",
                min_value=18,
                max_value=100,
            ),
            FeatureDefinition(
                name="credit_score",
                dtype="float",
                min_value=300,
                max_value=850,
            ),
            FeatureDefinition(
                name="segment",
                dtype="string",
                allowed_values=("prime", "subprime"),
            ),
        ),
    )


def test_validator_accepts_valid_dataframe() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [50000.0, 75000.0],
            "age": [35, 42],
            "credit_score": [720.0, 680.0],
            "segment": ["prime", "subprime"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is True
    assert result.errors == ()
    assert result.warnings == ()
    assert result.rows_checked == 2


def test_validator_rejects_missing_required_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [50000.0],
            "age": [35],
            "segment": ["prime"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert any("missing required feature: credit_score" in error for error in result.errors)


def test_validator_warns_about_unexpected_column() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [50000.0],
            "age": [35],
            "credit_score": [720.0],
            "segment": ["prime"],
            "extra_column": [123],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is True
    assert result.errors == ()
    assert "unexpected feature column: extra_column" in result.warnings


def test_validator_rejects_wrong_integer_dtype() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [50000.0],
            "age": [35.5],
            "credit_score": [720.0],
            "segment": ["prime"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert any(
        "feature 'age' must have dtype compatible with 'int'" in error for error in result.errors
    )


def test_validator_rejects_wrong_float_dtype() -> None:
    dataframe = pd.DataFrame(
        {
            "income": ["50000", "75000"],
            "age": [35, 42],
            "credit_score": [720.0, 680.0],
            "segment": ["prime", "subprime"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert any(
        "feature 'income' must have dtype compatible with 'float'" in error
        for error in result.errors
    )


def test_validator_rejects_null_non_nullable_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [50000.0, None],
            "age": [35, 42],
            "credit_score": [720.0, 680.0],
            "segment": ["prime", "subprime"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert any("feature 'income' contains 1 null value(s)" in error for error in result.errors)


def test_validator_rejects_value_below_minimum() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [-100.0, 50000.0],
            "age": [35, 42],
            "credit_score": [720.0, 680.0],
            "segment": ["prime", "subprime"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert any(
        "feature 'income' contains 1 value(s) below minimum 0" in error for error in result.errors
    )


def test_validator_rejects_value_above_maximum() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [50000.0, 75000.0],
            "age": [35, 101],
            "credit_score": [720.0, 680.0],
            "segment": ["prime", "subprime"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert any(
        "feature 'age' contains 1 value(s) above maximum 100" in error for error in result.errors
    )


def test_validator_rejects_disallowed_categorical_value() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [50000.0],
            "age": [35],
            "credit_score": [720.0],
            "segment": ["unknown"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert any(
        "feature 'segment' contains values outside allowed set" in error for error in result.errors
    )


def test_validator_collects_multiple_errors() -> None:
    dataframe = pd.DataFrame(
        {
            "income": [-100.0],
            "age": [101],
            "credit_score": [900.0],
            "segment": ["unknown"],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert result.error_count == 4


def test_validator_rejects_empty_dataframe() -> None:
    dataframe = pd.DataFrame(
        columns=[
            "income",
            "age",
            "credit_score",
            "segment",
        ]
    )

    result = FeatureValidator.validate(
        dataframe,
        _contract(),
    )

    assert result.valid is False
    assert result.rows_checked == 0
    assert "dataframe must not be empty" in result.errors


def test_validator_rejects_non_dataframe() -> None:
    result = FeatureValidator.validate(
        {"income": [50000.0]},
        _contract(),
    )

    assert result.valid is False
    assert result.rows_checked == 0
    assert result.errors == ("dataframe must be a pandas DataFrame",)
