from __future__ import annotations

import pytest

from ml.platform import (
    FeatureContract,
    FeatureDefinition,
    FeatureValidationResult,
)


def test_feature_definition_supports_constraints() -> None:
    feature = FeatureDefinition(
        name="credit_score",
        dtype="float32",
        required=True,
        nullable=False,
        min_value=300,
        max_value=850,
    )

    assert feature.name == "credit_score"
    assert feature.dtype == "float32"
    assert feature.required is True
    assert feature.nullable is False
    assert feature.min_value == 300
    assert feature.max_value == 850


def test_feature_definition_copies_allowed_values() -> None:
    values = ["A", "B"]

    feature = FeatureDefinition(
        name="segment",
        dtype="string",
        allowed_values=tuple(values),
    )

    values.append("C")

    assert feature.allowed_values == ("A", "B")


def test_feature_definition_rejects_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="feature name must not be empty",
    ):
        FeatureDefinition(
            name="",
            dtype="float32",
        )


def test_feature_definition_rejects_unsupported_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="unsupported feature dtype",
    ):
        FeatureDefinition(
            name="income",
            dtype="decimal128",
        )


def test_feature_definition_rejects_invalid_range() -> None:
    with pytest.raises(
        ValueError,
        match="min_value must not be greater",
    ):
        FeatureDefinition(
            name="age",
            dtype="int64",
            min_value=100,
            max_value=10,
        )


def test_feature_contract_preserves_feature_order() -> None:
    contract = FeatureContract(
        name="loan-default",
        features=(
            FeatureDefinition("income", "float32"),
            FeatureDefinition("age", "int64"),
            FeatureDefinition("credit_score", "float32"),
        ),
    )

    assert contract.feature_names == (
        "income",
        "age",
        "credit_score",
    )


def test_feature_contract_exposes_required_features() -> None:
    contract = FeatureContract(
        name="test-contract",
        features=(
            FeatureDefinition(
                "required_feature",
                "float32",
                required=True,
            ),
            FeatureDefinition(
                "optional_feature",
                "float32",
                required=False,
            ),
        ),
    )

    assert tuple(feature.name for feature in contract.required_features) == ("required_feature",)


def test_feature_contract_rejects_duplicate_names() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate feature names",
    ):
        FeatureContract(
            name="duplicate-contract",
            features=(
                FeatureDefinition("income", "float32"),
                FeatureDefinition("income", "float32"),
            ),
        )


def test_feature_validation_result_is_immutable() -> None:
    result = FeatureValidationResult(
        valid=False,
        errors=("missing feature",),
        warnings=("warning",),
        rows_checked=10,
    )

    assert result.valid is False
    assert result.errors == ("missing feature",)
    assert result.warnings == ("warning",)
    assert result.rows_checked == 10
    assert result.error_count == 1
    assert result.warning_count == 1

    with pytest.raises(AttributeError):
        result.valid = True
