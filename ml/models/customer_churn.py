from __future__ import annotations

from typing import Any, Final

import pandas as pd

from ml.platform import FeatureValidator
from .customer_churn_features import CUSTOMER_CHURN_FEATURE_CONTRACT

MODEL_NAME: Final[str] = "CustomerChurnModel"


FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "usage_hours",
    "payment_failures",
)


TARGET_COLUMN: Final[str] = "churn"


DEFAULT_MODEL_PARAMS: Final[dict[str, Any]] = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "liblinear",
    "class_weight": "balanced",
    "random_state": 42,
}


def validate_feature_columns(
    columns: list[str] | tuple[str, ...],
) -> None:
    """
    Validate that all required customer-churn features are present.
    """

    missing = [column for column in FEATURE_COLUMNS if column not in columns]

    if missing:
        raise ValueError(f"missing required customer-churn columns: {missing}")


def validate_feature_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate customer-churn data using the platform feature contract.
    """

    result = FeatureValidator.validate(
        dataframe,
        CUSTOMER_CHURN_FEATURE_CONTRACT,
    )

    if not result.valid:
        raise ValueError(
            "Customer Churn feature contract validation failed: " + "; ".join(result.errors)
        )
