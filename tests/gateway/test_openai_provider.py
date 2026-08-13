from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ai_platform.llm_gateway.providers.openai_provider import (
    OpenAIProvider,
)


@pytest.mark.asyncio
async def test_health_check():

    provider = OpenAIProvider()

    health = await provider.health_check()

    assert "status" in health
    assert "configured" in health
    assert "base_url" in health
    assert "default_model" in health


@pytest.mark.asyncio
async def test_list_models():

    provider = OpenAIProvider()

    models = await provider.list_models()

    assert "gpt-4o" in models
    assert "gpt-4.1" in models
    assert "openai-embedding" in models


@pytest.mark.asyncio
async def test_supported_chat_models():

    provider = OpenAIProvider()

    models = provider.supported_chat_models()

    assert "gpt-4o" in models


@pytest.mark.asyncio
async def test_supported_embedding_models():

    provider = OpenAIProvider()

    models = provider.supported_embedding_models()

    assert "openai-embedding" in models


@pytest.mark.asyncio
async def test_embeddings():

    provider = OpenAIProvider()

    vector = await provider.embeddings(
        {
            "model": "openai-embedding",
        }
    )

    assert vector == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_embeddings_success():

    provider = OpenAIProvider()

    fake_embedding = [0.01, 0.02, 0.03]

    fake_data = MagicMock()
    fake_data.embedding = fake_embedding

    fake_response = MagicMock()
    fake_response.data = [fake_data]

    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.embeddings(
        {
            "model": "openai-embedding",
            "text": "hello world",
        }
    )

    assert result == fake_embedding

    fake_client.embeddings.create.assert_awaited_once_with(
        model="text-embedding-3-small",
        input="hello world",
    )


@pytest.mark.asyncio
async def test_invalid_embedding_model():

    provider = OpenAIProvider()

    with pytest.raises(ValueError):
        await provider.embeddings(
            {
                "model": "bad-model",
            }
        )


@pytest.mark.asyncio
async def test_stream():

    provider = OpenAIProvider()

    class FakeEvent:
        def __init__(self, event_type, delta=None):
            self.type = event_type
            self.delta = delta

    async def fake_stream():
        yield FakeEvent(
            "response.output_text.delta",
            "openai-chunk1",
        )

        yield FakeEvent(
            "response.output_text.delta",
            "openai-chunk2",
        )

        yield FakeEvent(
            "response.completed",
        )

    fake_client = MagicMock()

    fake_client.responses.create = AsyncMock(return_value=fake_stream())

    provider.client = fake_client

    chunks = []

    async for chunk in provider.stream(
        {
            "prompt": "Hello",
            "model": "gpt-4o",
        }
    ):
        chunks.append(chunk)

    assert chunks == [
        "openai-chunk1",
        "openai-chunk2",
    ]

    fake_client.responses.create.assert_awaited_once_with(
        model="gpt-4o",
        input="Hello",
        stream=True,
    )


@pytest.mark.asyncio
async def test_chat_without_api_key():

    provider = OpenAIProvider()

    provider.client = None

    with pytest.raises(ProviderAuthenticationError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "gpt-4o",
            }
        )


@pytest.mark.asyncio
async def test_chat_success():

    provider = OpenAIProvider()

    fake_usage = MagicMock()
    fake_usage.input_tokens = 11
    fake_usage.output_tokens = 22

    fake_response = MagicMock()
    fake_response.output_text = "Hello from OpenAI"
    fake_response.usage = fake_usage

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "gpt-4o",
        }
    )

    assert result["reply"] == "Hello from OpenAI"
    assert result["usage"]["tokens_in"] == 11
    assert result["usage"]["tokens_out"] == 22


@pytest.mark.asyncio
async def test_chat_authentication_error():

    provider = OpenAIProvider()

    fake_client = MagicMock()

    fake_client.responses.create.side_effect = AuthenticationError(
        "bad key",
        response=MagicMock(),
        body={},
    )

    provider.client = fake_client

    with pytest.raises(ProviderAuthenticationError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "gpt-4o",
            }
        )


@pytest.mark.asyncio
async def test_chat_timeout():

    provider = OpenAIProvider()

    fake_client = MagicMock()

    fake_client.responses.create.side_effect = APITimeoutError(request=MagicMock())

    provider.client = fake_client

    with pytest.raises(ProviderTimeoutError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "gpt-4o",
            }
        )


@pytest.mark.asyncio
async def test_chat_connection_error():

    provider = OpenAIProvider()

    fake_client = MagicMock()

    fake_client.responses.create.side_effect = APIConnectionError(
        message="connection failed",
        request=MagicMock(),
    )

    provider.client = fake_client

    with pytest.raises(ProviderConnectionError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "gpt-4o",
            }
        )


@pytest.mark.asyncio
async def test_chat_success_without_usage():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.output_text = "Hello"
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "gpt-4o",
        }
    )

    assert result["reply"] == "Hello"
    assert result["usage"]["tokens_in"] == 0
    assert result["usage"]["tokens_out"] == 0


@pytest.mark.asyncio
async def test_chat_rate_limit_error():

    provider = OpenAIProvider()

    fake_client = MagicMock()

    request = httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
    )

    response = httpx.Response(
        status_code=429,
        request=request,
    )

    fake_client.responses.create.side_effect = RateLimitError(
        "quota exceeded",
        response=response,
        body={},
    )

    provider.client = fake_client

    with pytest.raises(ProviderRateLimitError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "gpt-4o",
            }
        )
