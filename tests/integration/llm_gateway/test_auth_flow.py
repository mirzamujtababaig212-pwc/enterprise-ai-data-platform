from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app


client = TestClient(app)


def test_missing_api_key():
    response = client.post(
        "/v1/chat",
        json={
            "provider": "openai",
            "model": "gpt-4.1",
            "prompt": "hello",
        },
    )

    assert response.status_code == 401


def test_invalid_api_key():
    response = client.post(
        "/v1/chat",
        headers={
            "x-api-key": "wrong-key",
        },
        json={
            "provider": "openai",
            "model": "gpt-4.1",
            "prompt": "hello",
        },
    )

    assert response.status_code == 401
