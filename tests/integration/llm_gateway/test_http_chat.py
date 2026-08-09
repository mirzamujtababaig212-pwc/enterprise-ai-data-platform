from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


def test_http_chat_non_streaming():
    response = client.post(
        "/v1/chat",
        json={
            "prompt": "Explain RAG in one sentence.",
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

    assert response.headers["content-type"].startswith("application/json")

    body = response.json()

    assert body is not None
