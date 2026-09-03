from __future__ import annotations

from typing import Any, Final


MODEL_NAME: Final[str] = "VehicleRiskModel"


FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "event_count",
    "avg_speed",
    "max_speed",
    "speed_stddev",
    "avg_rpm",
    "max_rpm",
    "avg_fuel_level",
    "min_fuel_level",
    "avg_battery",
    "avg_engine_temperature",
    "max_engine_temperature",
)


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
    """

    missing = [column for column in FEATURE_COLUMNS if column not in columns]

    if missing:
        raise ValueError(f"missing required vehicle-risk columns: {missing}")
