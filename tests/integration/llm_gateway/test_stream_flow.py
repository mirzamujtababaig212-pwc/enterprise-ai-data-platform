import pytest

from ai_platform.llm_gateway.routing.router import router


@pytest.mark.asyncio
async def test_stream():

    request = {
        "prompt": "Explain RAG.",
        "provider": "gemini",
        "model": "gemini-chat",
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True,
    }

    stream = router.route_stream(request)

    chunks = []

    async for chunk in stream:
        chunks.append(chunk)

    assert len(chunks) > 0

    assert all(isinstance(chunk, str) for chunk in chunks)
