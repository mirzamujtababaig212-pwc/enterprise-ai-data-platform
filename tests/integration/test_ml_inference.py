from __future__ import annotations

import pandas as pd
import pytest

from ml.inference import VehicleRiskPredictor


FEATURE_COLUMNS = [
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
]


@pytest.fixture
def vehicle_features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_count": 10,
                "avg_speed": 70.0,
                "max_speed": 80.0,
                "speed_stddev": 5.0,
                "avg_rpm": 1800.0,
                "max_rpm": 2500.0,
                "avg_fuel_level": 70.0,
                "min_fuel_level": 50.0,
                "avg_battery": 12.5,
                "avg_engine_temperature": 80.0,
                "max_engine_temperature": 90.0,
            }
        ]
    )


def test_champion_model_inference(
    vehicle_features: pd.DataFrame,
) -> None:
    predictor = VehicleRiskPredictor(
        model_name="VehicleRiskModel",
        model_alias="champion",
    )

    predictor.load()

    result = predictor.predict(vehicle_features)

    assert result.risk in (0, 1)

    assert result.model_name == "VehicleRiskModel"

    assert result.model_alias == "champion"

    if result.risk_probability is not None:
        assert 0.0 <= result.risk_probability <= 1.0


def test_batch_inference(
    vehicle_features: pd.DataFrame,
) -> None:
    predictor = VehicleRiskPredictor(
        model_name="VehicleRiskModel",
        model_alias="champion",
    )

    result = predictor.predict_batch(vehicle_features)

    assert len(result) == 1

    assert "risk" in result.columns

    assert "model_name" in result.columns

    assert "model_alias" in result.columns


def test_missing_feature_rejected() -> None:
    predictor = VehicleRiskPredictor(
        model_name="VehicleRiskModel",
        model_alias="champion",
    )

    incomplete = pd.DataFrame(
        [
            {
                "event_count": 10,
            }
        ]
    )

    with pytest.raises(ValueError):
        predictor.predict(incomplete)
