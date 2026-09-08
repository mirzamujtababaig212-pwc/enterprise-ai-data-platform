from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


def test_missing_api_key_returns_401():
    response = client.post(
        "/v1/chat",
        json={
            "prompt": "Explain RAG.",
            "provider": "gemini",
            "model": "gemini-chat",
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
        },
    )

    assert response.status_code == 401

    body = response.json()

    assert body["detail"] == "Invalid or missing API key"


def test_invalid_api_key_returns_401():
    response = client.post(
        "/v1/chat",
        json={
            "prompt": "Explain RAG.",
            "provider": "gemini",
            "model": "gemini-chat",
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False,
        },
        headers={
            "x-api-key": "wrong-key",
        },
    )

    assert response.status_code == 401

    body = response.json()

    assert body["detail"] == "Invalid or missing API key"


def test_valid_api_key_is_accepted():
    fake_response = {
        "reply": "RAG retrieves relevant context before generating an answer.",
        "usage": {
            "tokens_in": 5,
            "tokens_out": 8,
        },
    }

    with patch(
        "ai_platform.llm_gateway.api.main.router.route_chat",
        new=AsyncMock(return_value=fake_response),
    ):
        response = client.post(
            "/v1/chat",
            json={
                "prompt": "Explain RAG.",
                "provider": "gemini",
                "model": "gemini-chat",
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": False,
            },
            headers={
                "x-api-key": "super-secret-key",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["reply"] == fake_response["reply"]
    assert body["metrics"]["tokens_in"] == 5
    assert body["metrics"]["tokens_out"] == 8
    assert body["metrics"]["status"] == "success"
