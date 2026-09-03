from __future__ import annotations

import os

from fastapi.testclient import TestClient
from ai_platform.llm_gateway.api.main import app


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


def get_valid_api_key() -> str:
    api_key = os.getenv("API_KEY", "").strip()

    assert api_key, (
        "API_KEY must be configured in the test environment " "for authenticated integration tests."
    )

    return api_key


def get_registered_paths() -> set[str]:
    return set(app.openapi()["paths"].keys())


def test_vehicle_risk_route_is_registered() -> None:
    routes = get_registered_paths()

    assert "/api/v1/ml/vehicle-risk/predict" in routes


def test_vehicle_risk_requires_api_key() -> None:
    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=VALID_REQUEST,
    )

    assert response.status_code == 401

    payload = response.json()

    assert payload["detail"] == "Invalid or missing API key"


def test_vehicle_risk_rejects_invalid_api_key() -> None:
    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=VALID_REQUEST,
        headers={
            "X-API-Key": "definitely-invalid-api-key",
        },
    )

    assert response.status_code == 401

    payload = response.json()

    assert payload["detail"] == "Invalid or missing API key"


def test_vehicle_risk_accepts_valid_api_key() -> None:
    api_key = get_valid_api_key()

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=VALID_REQUEST,
        headers={
            "X-API-Key": api_key,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["model_name"] == "VehicleRiskModel"
    assert payload["model_alias"] == "champion"
    assert payload["risk"] in (0, 1)

    probability = payload["risk_probability"]

    assert probability is None or (0.0 <= probability <= 1.0)


def test_vehicle_risk_rejects_invalid_request() -> None:
    api_key = get_valid_api_key()

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json={
            "event_count": 10,
        },
        headers={
            "X-API-Key": api_key,
        },
    )

    assert response.status_code in (
        400,
        422,
    )


def test_vehicle_risk_rejects_unknown_request_shape() -> None:
    api_key = get_valid_api_key()

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json={
            "event_count": 10,
            "avg_speed": 70.0,
            "unexpected_field": "invalid",
        },
        headers={
            "X-API-Key": api_key,
        },
    )

    assert response.status_code in (
        400,
        422,
    )


def test_vehicle_risk_response_has_expected_schema() -> None:
    api_key = get_valid_api_key()

    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json=VALID_REQUEST,
        headers={
            "X-API-Key": api_key,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    expected_keys = {
        "risk",
        "risk_probability",
        "model_name",
        "model_alias",
    }

    assert set(payload.keys()) == expected_keys
