from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
from mlflow import MlflowClient

from ml.sagemaker.vehicle_risk import (
    input_fn,
    model_fn,
    output_fn,
    predict_fn,
)

FEATURE_VALUES = {
    "event_count": 12,
    "avg_speed": 55.0,
    "max_speed": 72.0,
    "speed_stddev": 6.0,
    "avg_rpm": 2100.0,
    "max_rpm": 2800.0,
    "avg_fuel_level": 72.0,
    "min_fuel_level": 61.0,
    "avg_battery": 88.0,
    "avg_engine_temperature": 91.0,
    "max_engine_temperature": 98.0,
}


def test_sagemaker_vehicle_risk_matches_mlflow_champion() -> None:
    client = MlflowClient()

    version = client.get_model_version_by_alias(
        "VehicleRiskModel",
        "champion",
    )

    assert version.name == "VehicleRiskModel"
    assert version.aliases == ["champion"]

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_dir = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=version.source,
                dst_path=temp_dir,
            )
        )

        assert (artifact_dir / "model.skops").is_file()
        assert (artifact_dir / "MLmodel").is_file()

        model = model_fn(artifact_dir)

        dataframe = input_fn(
            json.dumps(FEATURE_VALUES),
            "application/json",
        )

        prediction = predict_fn(
            dataframe,
            model,
        )

        response = output_fn(
            prediction,
            "application/json",
        )

    assert prediction["risk"] == 0
    assert prediction["risk_probability"] == 0.38
    assert prediction["model_name"] == "VehicleRiskModel"
    assert prediction["model_alias"] == "champion"

    assert json.loads(response) == prediction
