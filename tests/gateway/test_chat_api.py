from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)

HEADERS = {"x-api-key": "super-secret-key"}


def test_chat_endpoint_success():

    fake_response = {
        "reply": "Hello from mocked provider",
        "usage": {
            "tokens_in": 5,
            "tokens_out": 7,
        },
    }

    with patch(
        "ai_platform.llm_gateway.api.main.router.route_chat",
        new=AsyncMock(return_value=fake_response),
    ):

        response = client.post(
            "/v1/chat",
            headers=HEADERS,
            json={
                "provider": "openai",
                "model": "gpt-4o",
                "prompt": "Hello",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["reply"] == "Hello from mocked provider"

    assert body["metrics"]["tokens_in"] == 5

    assert body["metrics"]["tokens_out"] == 7

    assert body["metrics"]["status"] == "success"


def test_chat_without_api_key():

    response = client.post(
        "/v1/chat",
        json={
            "provider": "openai",
            "model": "gpt-4o",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 401


def test_chat_invalid_api_key():

    response = client.post(
        "/v1/chat",
        headers={"x-api-key": "wrong"},
        json={
            "provider": "openai",
            "model": "gpt-4o",
            "prompt": "Hello",
        },
    )

    assert response.status_code == 401


def test_health():

    fake_health = {
        "openai": {
            "status": "ok",
            "configured": True,
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
        }
    }

    with patch(
        "ai_platform.llm_gateway.api.main.router.route_health",
        new=AsyncMock(return_value=fake_health),
    ):

        response = client.get(
            "/v1/health",
            headers=HEADERS,
        )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"

    assert body["providers"]["openai"]["status"] == "ok"

    assert body["providers"]["openai"]["configured"] is True


def test_models():

    with patch(
        "ai_platform.llm_gateway.api.main.router.route_models",
        new=AsyncMock(
            return_value={
                "openai": [
                    "gpt-4o",
                    "gpt-4.1",
                ]
            }
        ),
    ):

        response = client.get(
            "/v1/models",
            headers=HEADERS,
        )

    assert response.status_code == 200

    body = response.json()

    assert "openai" in body
