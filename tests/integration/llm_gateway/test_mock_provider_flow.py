"""End-to-end tests for the deterministic mock provider."""

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app


client = TestClient(app)


API_KEY = "super-secret-key"


def test_mock_chat_requires_authentication() -> None:
    """Chat endpoint must reject requests without an API key."""

    response = client.post(
        "/v1/chat",
        json={
            "provider": "mock",
            "model": "mock-gpt",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 401


def test_mock_chat_rejects_invalid_authentication() -> None:
    """Chat endpoint must reject an invalid API key."""

    response = client.post(
        "/v1/chat",
        headers={
            "x-api-key": "invalid-key",
        },
        json={
            "provider": "mock",
            "model": "mock-gpt",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 401


def test_mock_chat_success() -> None:
    """Full gateway path must successfully execute the mock provider."""

    response = client.post(
        "/v1/chat",
        headers={
            "x-api-key": API_KEY,
        },
        json={
            "provider": "mock",
            "model": "mock-gpt",
            "prompt": "Hello Enterprise AI",
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert "reply" in payload
    assert "metrics" in payload

    assert "Mock response" in payload["reply"]

    metrics = payload["metrics"]

    assert metrics["request_id"]
    assert metrics["timestamp"]
    assert metrics["latency_ms"] >= 0
    assert metrics["tokens_in"] >= 0
    assert metrics["tokens_out"] >= 0
    assert metrics["estimated_cost"] >= 0
    assert metrics["status"] == "success"


def test_mock_models_are_exposed() -> None:
    """Mock models must appear in the model registry/API."""

    response = client.get(
        "/v1/models",
        headers={
            "x-api-key": API_KEY,
        },
    )

    assert response.status_code == 200

    models = response.json()

    assert "mock" in models
    assert "mock-gpt" in models["mock"]


def test_mock_health() -> None:
    """Gateway health endpoint must remain operational."""

    response = client.get(
        "/v1/health",
        headers={
            "x-api-key": API_KEY,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"]
    assert "providers" in payload
