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
