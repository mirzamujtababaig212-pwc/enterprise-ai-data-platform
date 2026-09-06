from __future__ import annotations

import pandas as pd
import pytest

from ml.models.loan_default import validate_feature_dataframe
from ml.models.loan_default_features import (
    LOAN_DEFAULT_FEATURE_CONTRACT,
)


def _valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [60000.0, 75000.0],
            "age": [35, 42],
            "credit_score": [720.0, 680.0],
            "loan_amount": [20000.0, 30000.0],
            "employment_years": [8.0, 12.0],
            "debt_to_income": [0.25, 0.40],
        }
    )


def test_loan_default_feature_contract_definition() -> None:
    assert LOAN_DEFAULT_FEATURE_CONTRACT.name == "loan-default"

    assert LOAN_DEFAULT_FEATURE_CONTRACT.feature_names == (
        "income",
        "age",
        "credit_score",
        "loan_amount",
        "employment_years",
        "debt_to_income",
    )


def test_loan_default_feature_contract_accepts_valid_dataframe() -> None:
    validate_feature_dataframe(_valid_dataframe())


def test_loan_default_rejects_age_below_minimum() -> None:
    dataframe = _valid_dataframe()
    dataframe["age"] = 17

    with pytest.raises(
        ValueError,
        match="age.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_age_above_maximum() -> None:
    dataframe = _valid_dataframe()
    dataframe["age"] = 101

    with pytest.raises(
        ValueError,
        match="age.*above maximum",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_credit_score_below_minimum() -> None:
    dataframe = _valid_dataframe()
    dataframe["credit_score"] = 299.0

    with pytest.raises(
        ValueError,
        match="credit_score.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_credit_score_above_maximum() -> None:
    dataframe = _valid_dataframe()
    dataframe["credit_score"] = 851.0

    with pytest.raises(
        ValueError,
        match="credit_score.*above maximum",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_debt_to_income_above_maximum() -> None:
    dataframe = _valid_dataframe()
    dataframe["debt_to_income"] = 1.01

    with pytest.raises(
        ValueError,
        match="debt_to_income.*above maximum",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_negative_income() -> None:
    dataframe = _valid_dataframe()
    dataframe["income"] = -1.0

    with pytest.raises(
        ValueError,
        match="income.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_negative_loan_amount() -> None:
    dataframe = _valid_dataframe()
    dataframe["loan_amount"] = -1.0

    with pytest.raises(
        ValueError,
        match="loan_amount.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_null_feature() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "credit_score"] = None

    with pytest.raises(
        ValueError,
        match="credit_score.*null value",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_wrong_integer_dtype() -> None:
    dataframe = _valid_dataframe()
    dataframe["age"] = [35.5, 42.5]

    with pytest.raises(
        ValueError,
        match="age.*int",
    ):
        validate_feature_dataframe(dataframe)


def test_loan_default_rejects_wrong_numeric_dtype() -> None:
    dataframe = _valid_dataframe()
    dataframe["income"] = ["60000", "75000"]

    with pytest.raises(
        ValueError,
        match="income.*float",
    ):
        validate_feature_dataframe(dataframe)
