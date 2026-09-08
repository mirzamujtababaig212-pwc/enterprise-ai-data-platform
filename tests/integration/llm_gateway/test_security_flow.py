from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app
from ai_platform.llm_gateway.config.settings import settings


VALID_API_KEY = "super-secret-key"

CHAT_PAYLOAD = {
    "prompt": "Hello",
    "provider": "mock",
    "model": "mock-gpt",
}


def test_missing_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", VALID_API_KEY)

    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        json=CHAT_PAYLOAD,
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid or missing API key",
    }


def test_invalid_api_key_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", VALID_API_KEY)

    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        json=CHAT_PAYLOAD,
        headers={
            "x-api-key": "invalid-key",
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid or missing API key",
    }


def test_valid_api_key_is_accepted(monkeypatch):
    monkeypatch.setattr(settings, "API_KEY", VALID_API_KEY)

    client = TestClient(app)

    response = client.post(
        "/v1/chat",
        json=CHAT_PAYLOAD,
        headers={
            "x-api-key": VALID_API_KEY,
        },
    )

    assert response.status_code == 200


def test_health_endpoint_does_not_require_api_key():
    client = TestClient(app)

    response = client.get("/v1/health")

    assert response.status_code == 200


def test_metrics_endpoint_does_not_require_api_key():
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200


def test_openapi_endpoint_does_not_require_api_key():
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200


def test_docs_endpoint_does_not_require_api_key():
    client = TestClient(app)

    response = client.get("/docs")

    assert response.status_code == 200


def test_redoc_endpoint_does_not_require_api_key():
    client = TestClient(app)

    response = client.get("/redoc")

    assert response.status_code == 200


def test_favicon_endpoint_does_not_require_api_key():
    client = TestClient(app)

    response = client.get("/favicon.ico")

    assert response.status_code == 404
