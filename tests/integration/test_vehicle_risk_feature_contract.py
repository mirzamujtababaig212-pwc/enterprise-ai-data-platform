from __future__ import annotations

import pandas as pd

from ml.models.vehicle_risk_features import (
    VEHICLE_RISK_FEATURE_CONTRACT,
)
from ml.platform import FeatureValidator


def test_vehicle_risk_contract_defines_expected_features() -> None:
    assert VEHICLE_RISK_FEATURE_CONTRACT.name == "vehicle-risk"

    assert VEHICLE_RISK_FEATURE_CONTRACT.feature_names == (
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


def test_vehicle_risk_contract_accepts_valid_features() -> None:
    dataframe = pd.DataFrame(
        {
            "event_count": [10, 20],
            "avg_speed": [55.0, 70.0],
            "max_speed": [90.0, 120.0],
            "speed_stddev": [5.0, 8.0],
            "avg_rpm": [1800.0, 2200.0],
            "max_rpm": [3000.0, 4000.0],
            "avg_fuel_level": [70.0, 50.0],
            "min_fuel_level": [40.0, 20.0],
            "avg_battery": [12.5, 12.2],
            "avg_engine_temperature": [85.0, 95.0],
            "max_engine_temperature": [98.0, 104.0],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        VEHICLE_RISK_FEATURE_CONTRACT,
    )

    assert result.valid is True
    assert result.errors == ()
    assert result.warnings == ()


def test_vehicle_risk_contract_rejects_invalid_ranges() -> None:
    dataframe = pd.DataFrame(
        {
            "event_count": [10],
            "avg_speed": [55.0],
            "max_speed": [90.0],
            "speed_stddev": [5.0],
            "avg_rpm": [1800.0],
            "max_rpm": [3000.0],
            "avg_fuel_level": [110.0],
            "min_fuel_level": [-5.0],
            "avg_battery": [12.5],
            "avg_engine_temperature": [85.0],
            "max_engine_temperature": [98.0],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        VEHICLE_RISK_FEATURE_CONTRACT,
    )

    assert result.valid is False
    assert any("avg_fuel_level" in error for error in result.errors)
    assert any("min_fuel_level" in error for error in result.errors)


def test_vehicle_risk_contract_detects_missing_feature() -> None:
    dataframe = pd.DataFrame(
        {
            "event_count": [10],
            "avg_speed": [55.0],
            "max_speed": [90.0],
            "speed_stddev": [5.0],
            "avg_rpm": [1800.0],
            "max_rpm": [3000.0],
            "avg_fuel_level": [70.0],
            "min_fuel_level": [40.0],
            "avg_battery": [12.5],
            "avg_engine_temperature": [85.0],
        }
    )

    result = FeatureValidator.validate(
        dataframe,
        VEHICLE_RISK_FEATURE_CONTRACT,
    )

    assert result.valid is False
    assert any(
        "missing required feature: max_engine_temperature" in error for error in result.errors
    )
