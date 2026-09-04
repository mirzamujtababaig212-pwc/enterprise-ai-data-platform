from fastapi.testclient import TestClient

from app.control_plane.app import app

client = TestClient(app)

AUTH_HEADERS = {"x-api-key": "super-secret-key"}


def test_vehicle_risk_route_rejects_missing_fields() -> None:
    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json={},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_vehicle_risk_route_rejects_unknown_fields() -> None:
    response = client.post(
        "/api/v1/ml/vehicle-risk/predict",
        json={
            "event_count": 1,
            "avg_speed": 10.0,
            "max_speed": 20.0,
            "speed_stddev": 2.0,
            "avg_rpm": 1000.0,
            "max_rpm": 2000.0,
            "avg_fuel_level": 80.0,
            "min_fuel_level": 70.0,
            "avg_battery": 12.5,
            "avg_engine_temperature": 80.0,
            "max_engine_temperature": 90.0,
            "unexpected": "field",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422
