from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ml.api.routes import router

app = FastAPI()
app.include_router(router)

client = TestClient(app)


VALID_PAYLOAD = {
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


@pytest.fixture(autouse=True)
def mock_vehicle_risk_mlflow_model():
    """Mock MLflow model loading for host-side API integration tests."""
    mock_model = MagicMock()

    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    with patch(
        "mlflow.sklearn.load_model",
        return_value=mock_model,
    ) as mock_load_model:
        yield mock_load_model


def test_vehicle_risk_api() -> None:
    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=VALID_PAYLOAD,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["risk"] in (0, 1)

    assert body["model_name"] == "VehicleRiskModel"

    assert body["model_alias"] == "champion"

    if body["risk_probability"] is not None:
        assert 0.0 <= body["risk_probability"] <= 1.0


def test_vehicle_risk_api_rejects_missing_feature() -> None:
    payload = VALID_PAYLOAD.copy()

    del payload["avg_speed"]

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_vehicle_risk_api_rejects_unknown_feature() -> None:
    payload = VALID_PAYLOAD.copy()

    payload["unknown_feature"] = 123

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=payload,
    )

    assert response.status_code == 422
