from __future__ import annotations

from typing import Final

from .features import FeatureContract, FeatureDefinition

VEHICLE_RISK_MODEL_NAME: Final[str] = "VehicleRiskModel"

VEHICLE_RISK_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
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

VEHICLE_RISK_FEATURE_CONTRACT = FeatureContract(
    name="vehicle-risk",
    version="v1",
    features=(
        FeatureDefinition(
            name="event_count",
            dtype="int",
            min_value=0,
        ),
        FeatureDefinition(
            name="avg_speed",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="max_speed",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="speed_stddev",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="avg_rpm",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="max_rpm",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="avg_fuel_level",
            dtype="float",
            min_value=0,
            max_value=100,
        ),
        FeatureDefinition(
            name="min_fuel_level",
            dtype="float",
            min_value=0,
            max_value=100,
        ),
        FeatureDefinition(
            name="avg_battery",
            dtype="float",
            min_value=0,
        ),
        FeatureDefinition(
            name="avg_engine_temperature",
            dtype="float",
        ),
        FeatureDefinition(
            name="max_engine_temperature",
            dtype="float",
        ),
    ),
)

__all__ = [
    "VEHICLE_RISK_FEATURE_COLUMNS",
    "VEHICLE_RISK_FEATURE_CONTRACT",
    "VEHICLE_RISK_MODEL_NAME",
]
