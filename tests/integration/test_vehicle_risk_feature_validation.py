from __future__ import annotations

import pandas as pd
import pytest

from ml.models.vehicle_risk import validate_feature_dataframe


def _valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_count": [10],
            "avg_speed": [50.0],
            "max_speed": [90.0],
            "speed_stddev": [2.0],
            "avg_rpm": [1500.0],
            "max_rpm": [1800.0],
            "avg_fuel_level": [80.0],
            "min_fuel_level": [70.0],
            "avg_battery": [12.5],
            "avg_engine_temperature": [80.0],
            "max_engine_temperature": [90.0],
        }
    )


def test_vehicle_risk_dataframe_validation_accepts_valid_data() -> None:
    validate_feature_dataframe(_valid_dataframe())


def test_vehicle_risk_dataframe_validation_rejects_invalid_dtype() -> None:
    dataframe = _valid_dataframe()
    dataframe["event_count"] = [10.5]

    with pytest.raises(
        ValueError,
        match="event_count.*compatible with 'int'",
    ):
        validate_feature_dataframe(dataframe)


def test_vehicle_risk_dataframe_validation_rejects_null() -> None:
    dataframe = _valid_dataframe()
    dataframe["avg_speed"] = [None]

    with pytest.raises(
        ValueError,
        match="avg_speed.*null",
    ):
        validate_feature_dataframe(dataframe)


def test_vehicle_risk_dataframe_validation_rejects_range_violation() -> None:
    dataframe = _valid_dataframe()
    dataframe["min_fuel_level"] = [-1.0]

    with pytest.raises(
        ValueError,
        match="min_fuel_level.*below minimum",
    ):
        validate_feature_dataframe(dataframe)


def test_vehicle_risk_dataframe_validation_allows_unexpected_column() -> None:
    dataframe = _valid_dataframe()
    dataframe["telemetry_source"] = ["test"]

    # Unexpected columns are warnings, not validation failures.
    validate_feature_dataframe(dataframe)
