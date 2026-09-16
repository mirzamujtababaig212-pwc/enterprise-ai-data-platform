from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ml.sagemaker import server

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


def test_ping_returns_200_when_model_is_loaded(monkeypatch):
    monkeypatch.setattr(server, "MODEL", object())

    client = TestClient(server.app)

    response = client.get("/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ping_returns_503_when_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr(server, "MODEL", None)

    client = TestClient(server.app)

    response = client.get("/ping")

    assert response.status_code == 503
    assert response.json() == {"detail": "model is not loaded"}


def test_invocations_returns_vehicle_risk_prediction(monkeypatch):
    expected = {
        "risk": 0,
        "risk_probability": 0.38,
        "model_name": "VehicleRiskModel",
        "model_alias": "champion",
    }

    monkeypatch.setattr(server, "MODEL", object())
    monkeypatch.setattr(
        server,
        "predict_fn",
        lambda dataframe, model: expected,
    )

    client = TestClient(server.app)

    response = client.post(
        "/invocations",
        content=json.dumps(FEATURE_VALUES),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_invocations_rejects_unsupported_content_type(monkeypatch):
    monkeypatch.setattr(server, "MODEL", object())

    client = TestClient(server.app)

    response = client.post(
        "/invocations",
        content=json.dumps(FEATURE_VALUES),
        headers={"Content-Type": "text/plain"},
    )

    assert response.status_code == 400
    assert "Unsupported content type" in response.json()["detail"]


def test_invocations_returns_503_when_model_is_not_loaded(monkeypatch):
    monkeypatch.setattr(server, "MODEL", None)

    client = TestClient(server.app)

    response = client.post(
        "/invocations",
        content=json.dumps(FEATURE_VALUES),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "model is not loaded"}
