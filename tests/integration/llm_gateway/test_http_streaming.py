from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)

HEADERS = {"x-api-key": "super-secret-key"}


async def fake_gemini_stream(request):
    yield "RAG retrieves relevant context."
    yield " The LLM uses that context to answer the question."


def test_http_streaming():
    with patch(
        "ai_platform.llm_gateway.api.main.router.route_stream",
        new=fake_gemini_stream,
    ):
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
            headers=HEADERS,
        )

    assert response.status_code == 200

    content_type = response.headers.get(
        "content-type",
        "",
    )

    assert content_type.startswith("text/event-stream")

    body = response.text

    assert "data: RAG retrieves relevant context.\n\n" in body
    assert "data:  The LLM uses that context to answer the question.\n\n" in body
    assert "data: [DONE]\n\n" in body

    first_chunk = "data: RAG retrieves relevant context.\n\n"
    second_chunk = "data:  The LLM uses that context to answer the question.\n\n"
    done = "data: [DONE]\n\n"

    assert body.index(first_chunk) < body.index(second_chunk)
    assert body.index(second_chunk) < body.index(done)
