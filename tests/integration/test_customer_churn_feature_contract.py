from __future__ import annotations

import pandas as pd
import pytest

from ml.models.customer_churn import validate_feature_dataframe
from ml.models.customer_churn_features import (
    CUSTOMER_CHURN_FEATURE_CONTRACT,
)


def _valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tenure_months": [12, 24],
            "monthly_charges": [75.0, 95.0],
            "total_charges": [900.0, 2280.0],
            "support_tickets": [2, 4],
            "usage_hours": [120.0, 180.0],
            "payment_failures": [0, 1],
        }
    )


def test_customer_churn_feature_contract_definition() -> None:
    assert CUSTOMER_CHURN_FEATURE_CONTRACT.name == "customer-churn"

    assert CUSTOMER_CHURN_FEATURE_CONTRACT.feature_names == (
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "support_tickets",
        "usage_hours",
        "payment_failures",
    )


def test_customer_churn_feature_contract_accepts_valid_dataframe() -> None:
    dataframe = _valid_dataframe()

    validate_feature_dataframe(dataframe)


def test_customer_churn_feature_contract_rejects_negative_tenure() -> None:
    dataframe = _valid_dataframe()
    dataframe["tenure_months"] = -1

    with pytest.raises(
        ValueError,
        match="tenure_months.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_customer_churn_feature_contract_rejects_negative_support_tickets() -> None:
    dataframe = _valid_dataframe()
    dataframe["support_tickets"] = -1

    with pytest.raises(
        ValueError,
        match="support_tickets.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_customer_churn_feature_contract_rejects_negative_payment_failures() -> None:
    dataframe = _valid_dataframe()
    dataframe["payment_failures"] = -1

    with pytest.raises(
        ValueError,
        match="payment_failures.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_customer_churn_feature_contract_rejects_negative_monthly_charges() -> None:
    dataframe = _valid_dataframe()
    dataframe["monthly_charges"] = -1.0

    with pytest.raises(
        ValueError,
        match="monthly_charges.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_customer_churn_feature_contract_rejects_nulls() -> None:
    dataframe = _valid_dataframe()
    dataframe.loc[0, "usage_hours"] = None

    with pytest.raises(
        ValueError,
        match="usage_hours.*null value",
    ):
        validate_feature_dataframe(dataframe)


def test_customer_churn_feature_contract_rejects_wrong_dtype() -> None:
    dataframe = _valid_dataframe()
    dataframe["support_tickets"] = [1.5, 2.5]

    with pytest.raises(
        ValueError,
        match="support_tickets.*int",
    ):
        validate_feature_dataframe(dataframe)
