from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app
from ai_platform.llm_gateway.config.settings import settings


client = TestClient(app)


VALID_REQUEST = {
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


def _api_headers() -> dict[str, str]:
    """Return authentication headers using the configured API key."""
    api_key = getattr(settings, "API_KEY", None)

    if not api_key:
        api_key = getattr(settings, "api_key", None)

    if not api_key:
        return {}

    first_key = next(
        (key.strip() for key in str(api_key).split(",") if key.strip()),
        "",
    )

    if not first_key:
        return {}

    return {
        "X-API-Key": first_key,
    }


@pytest.fixture(autouse=True)
def mock_vehicle_risk_mlflow_model() -> MagicMock:
    """
    Mock MLflow model loading for host-side API integration tests.

    These tests verify the FastAPI route, authentication, request validation,
    response structure, and predictor integration.

    They must not require a live MLflow/MinIO/S3 environment.
    """
    mock_model = MagicMock()

    # Binary classifier prediction.
    mock_model.predict.return_value = np.array([0])

    # Probability for classes [0, 1].
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    with patch(
        "ml.inference.vehicle_risk.mlflow.sklearn.load_model",
        return_value=mock_model,
    ):
        yield mock_model


def test_vehicle_risk_router_is_registered() -> None:
    """Verify that the vehicle-risk route is registered in FastAPI."""
    routes = app.openapi()["paths"]

    assert "/api/v1/ml/vehicle-risk/predict" in routes
    assert "post" in routes["/api/v1/ml/vehicle-risk/predict"]


def test_vehicle_risk_endpoint_is_available() -> None:
    """Verify a valid vehicle-risk request returns a successful response."""
    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=VALID_REQUEST,
        headers=_api_headers(),
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["model_name"] == "VehicleRiskModel"
    assert payload["model_alias"] == "champion"
    assert payload["risk"] in (0, 1)

    probability = payload["risk_probability"]

    assert probability is None or (0.0 <= probability <= 1.0)


def test_vehicle_risk_endpoint_rejects_invalid_input() -> None:
    """Verify invalid vehicle-risk input is rejected."""
    invalid_request = VALID_REQUEST.copy()
    invalid_request["event_count"] = "not_a_number"

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=invalid_request,
        headers=_api_headers(),
    )

    assert response.status_code in (400, 422)


def test_vehicle_risk_endpoint_rejects_unknown_request_shape() -> None:
    """Verify unexpected request fields are rejected."""
    invalid_request = VALID_REQUEST.copy()
    invalid_request["unexpected_field"] = "invalid"

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=invalid_request,
        headers=_api_headers(),
    )

    assert response.status_code in (400, 422)


def test_vehicle_risk_endpoint_rejects_missing_api_key() -> None:
    """Verify requests without an API key are rejected."""
    if not _api_headers():
        pytest.skip("API key authentication is not configured in the current test environment.")

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=VALID_REQUEST,
    )

    assert response.status_code in (401, 403)
