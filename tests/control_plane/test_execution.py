from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from app.control_plane.app import app

client = TestClient(app)

AUTH_HEADERS = {"x-api-key": "super-secret-key"}


def test_control_plane_chat_executes_mock_provider(usage_store) -> None:
    response = client.post(
        "/api/v1/llm/chat",
        json={
            "prompt": "Hello Enterprise AI.",
            "provider": "mock",
            "model": "mock-gpt",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["reply"] == "Mock response: Hello Enterprise AI."
    assert payload["metrics"]["status"] == "success"
    assert payload["metrics"]["tokens_in"] > 0
    assert payload["metrics"]["tokens_out"] > 0


def test_control_plane_embeddings_executes_mock_provider(usage_store) -> None:
    response = client.post(
        "/api/v1/llm/embeddings",
        json={
            "provider": "mock",
            "model": "mock-embedding",
            "text": "Enterprise AI Platform",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200

    payload = response.json()

    assert isinstance(payload["vector"], list)
    assert len(payload["vector"]) > 0
    assert payload["metrics"]["status"] == "success"
    assert payload["metrics"]["tokens_in"] > 0


def test_control_plane_vehicle_risk_executes_champion_model() -> None:
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    with patch(
        "mlflow.sklearn.load_model",
        return_value=mock_model,
    ):
        response = client.post(
            "/api/v1/ml/vehicle-risk/predict",
            json={
                "event_count": 10,
                "avg_speed": 55.0,
                "max_speed": 85.0,
                "speed_stddev": 12.0,
                "avg_rpm": 2200.0,
                "max_rpm": 3500.0,
                "avg_fuel_level": 65.0,
                "min_fuel_level": 45.0,
                "avg_battery": 12.4,
                "avg_engine_temperature": 88.0,
                "max_engine_temperature": 102.0,
            },
            headers=AUTH_HEADERS,
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["risk"] in (0, 1)
    assert 0.0 <= payload["risk_probability"] <= 1.0
    assert payload["model_alias"] == "champion"
    assert payload["model_name"]
