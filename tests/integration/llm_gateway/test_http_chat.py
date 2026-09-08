from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)

HEADERS = {"x-api-key": "super-secret-key"}


def test_http_chat_non_streaming():
    fake_response = {
        "reply": "RAG retrieves relevant context and uses it to ground an LLM response.",
        "usage": {
            "tokens_in": 8,
            "tokens_out": 14,
        },
    }

    with patch(
        "ai_platform.llm_gateway.api.main.router.route_chat",
        new=AsyncMock(return_value=fake_response),
    ) as mock_route_chat:
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
            headers=HEADERS,
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()

    assert body["reply"] == fake_response["reply"]
    assert body["metrics"]["tokens_in"] == 8
    assert body["metrics"]["tokens_out"] == 14
    assert body["metrics"]["status"] == "success"

    mock_route_chat.assert_awaited_once()
