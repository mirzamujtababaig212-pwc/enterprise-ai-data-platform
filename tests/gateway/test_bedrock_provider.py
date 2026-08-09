import pytest

from ai_platform.llm_gateway.providers.bedrock_provider import (
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    BedrockProvider,
)


@pytest.fixture
def provider():
    return BedrockProvider()


@pytest.mark.asyncio
async def test_chat(provider):
    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "bedrock-chat",
        }
    )

    assert result == {"reply": "Bedrock echo: Hello"}


@pytest.mark.asyncio
async def test_stream(provider):
    chunks = []
    async for chunk in provider.stream({}):
        chunks.append(chunk)
    assert chunks == [
        "bedrock-chunk1",
        "bedrock-chunk2",
    ]


@pytest.mark.asyncio
async def test_embeddings(provider):
    result = await provider.embeddings({"model": "bedrock-embedding"})

    assert result == [1.0, 1.1, 1.2]


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
        "bedrock-chat",
        "bedrock-embedding",
    ]


def test_supported_chat_models(provider):
    assert set(provider.supported_chat_models()) == SUPPORTED_CHAT_MODELS


def test_supported_embedding_models(provider):
    assert set(provider.supported_embedding_models()) == SUPPORTED_EMBEDDING_MODELS
