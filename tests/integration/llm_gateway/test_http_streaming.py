from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


def test_http_streaming():
    response = client.post(
        "/v1/chat",
        json={
            "prompt": "Explain RAG in one sentence.",
            "provider": "gemini",
            "model": "gemini-chat",
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
        },
        headers={
            "x-api-key": "super-secret-key",
        },
    )

    assert response.status_code == 200

    content_type = response.headers.get(
        "content-type",
        "",
    )

    assert content_type.startswith("text/event-stream")

    body = response.text

    assert "data: gemini-chunk1" in body
    assert "data: gemini-chunk2" in body
    assert "data: [DONE]" in body

    assert body.index("data: gemini-chunk1") < body.index("data: gemini-chunk2")

    assert body.index("data: gemini-chunk2") < body.index("data: [DONE]")
