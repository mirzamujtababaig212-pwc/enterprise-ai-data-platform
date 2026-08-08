import pytest

from ai_platform.llm_gateway.providers.gemini_provider import (
    GeminiProvider,
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
)


@pytest.fixture
def provider():
    return GeminiProvider()


@pytest.mark.asyncio
async def test_chat(provider):
    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "gemini-chat",
        }
    )

    assert result == {"reply": "Gemini echo: Hello"}


@pytest.mark.asyncio
async def test_stream(provider):
    chunks = []
    async for chunk in provider.stream({}):
        chunks.append(chunk)
    assert chunks == [
        "gemini-chunk1",
        "gemini-chunk2",
    ]


@pytest.mark.asyncio
async def test_embeddings(provider):
    result = await provider.embeddings({"model": "gemini-embedding"})

    assert result == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_embeddings_invalid_model(provider):
    with pytest.raises(ValueError):
        await provider.embeddings({"model": "bad-model"})


@pytest.mark.asyncio
async def test_health(provider):
    assert await provider.health_check() == {"status": "ok"}


@pytest.mark.asyncio
async def test_models(provider):
    assert await provider.list_models() == [
        "gemini-chat",
        "gemini-embedding",
    ]


def test_supported_chat_models(provider):
    assert set(provider.supported_chat_models()) == SUPPORTED_CHAT_MODELS


def test_supported_embedding_models(provider):
    assert set(provider.supported_embedding_models()) == SUPPORTED_EMBEDDING_MODELS
