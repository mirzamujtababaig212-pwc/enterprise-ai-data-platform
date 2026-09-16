from __future__ import annotations

from typing import Any, Final

import pandas as pd

from ml.contracts import FeatureValidator
from ml.contracts.vehicle_risk import (
    VEHICLE_RISK_FEATURE_COLUMNS,
    VEHICLE_RISK_FEATURE_CONTRACT,
    VEHICLE_RISK_MODEL_NAME,
)

MODEL_NAME: Final[str] = VEHICLE_RISK_MODEL_NAME
FEATURE_COLUMNS: Final[tuple[str, ...]] = VEHICLE_RISK_FEATURE_COLUMNS


TARGET_COLUMN: Final[str] = "risk"


DEFAULT_MODEL_PARAMS: Final[dict[str, Any]] = {
    "n_estimators": 100,
    "random_state": 42,
    "class_weight": "balanced",
}


def validate_feature_columns(
    columns: list[str] | tuple[str, ...],
) -> None:
    """
    Validate that all required vehicle-risk features are present.

    This legacy compatibility function preserves the existing public API.
    """

    missing = [column for column in FEATURE_COLUMNS if column not in columns]

    if missing:
        raise ValueError(f"missing required vehicle-risk columns: {missing}")


def validate_feature_dataframe(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate vehicle-risk data using the platform feature contract.
    """

    result = FeatureValidator.validate(
        dataframe,
        VEHICLE_RISK_FEATURE_CONTRACT,
    )

    if not result.valid:
        raise ValueError(
            "Vehicle Risk feature contract validation failed: " + "; ".join(result.errors)
        )
