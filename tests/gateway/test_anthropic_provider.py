import pytest

from ai_platform.llm_gateway.providers.anthropic_provider import (
    AnthropicProvider,
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
)


@pytest.fixture
def provider():
    return AnthropicProvider()


@pytest.mark.asyncio
async def test_chat(provider):
    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "anthropic-chat",
        }
    )

    assert result == {"reply": "Anthropic echo: Hello"}


@pytest.mark.asyncio
async def test_stream(provider):
    chunks = []
    async for chunk in provider.stream({}):
        chunks.append(chunk)
    assert chunks == [
        "anthropic-chunk1",
        "anthropic-chunk2",
    ]


@pytest.mark.asyncio
async def test_embeddings(provider):
    result = await provider.embeddings({"model": "anthropic-embedding"})

    assert result == [0.7, 0.8, 0.9]


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
        "anthropic-chat",
        "anthropic-embedding",
    ]


def test_supported_chat_models(provider):
    assert set(provider.supported_chat_models()) == SUPPORTED_CHAT_MODELS


def test_supported_embedding_models(provider):
    assert set(provider.supported_embedding_models()) == SUPPORTED_EMBEDDING_MODELS
