from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import boto3
import mlflow
import pytest
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
    if os.getenv("RUN_AWS_INTEGRATION") == "1":
        pytest.skip("Local MLflow/MinIO parity test")

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


@pytest.mark.aws
def test_sagemaker_vehicle_risk_aws_endpoint() -> None:
    if os.getenv("RUN_AWS_INTEGRATION") != "1":
        pytest.skip("Set RUN_AWS_INTEGRATION=1 to run the AWS integration test")

    runtime = boto3.client(
        "sagemaker-runtime",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )

    response = runtime.invoke_endpoint(
        EndpointName="enterprise-ai-platform-dev-vehicle-risk",
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(
            {
                "event_count": 3,
                "avg_speed": 48.5,
                "max_speed": 72.0,
                "speed_stddev": 8.2,
                "avg_rpm": 1850.0,
                "max_rpm": 2400.0,
                "avg_fuel_level": 62.0,
                "min_fuel_level": 58.0,
                "avg_battery": 13.8,
                "avg_engine_temperature": 91.0,
                "max_engine_temperature": 96.0,
            }
        ),
    )

    prediction = json.loads(response["Body"].read())

    assert prediction["risk"] == 0
    assert prediction["risk_probability"] == 0.48
    assert prediction["model_name"] == "VehicleRiskModel"
    assert prediction["model_alias"] == "champion"
