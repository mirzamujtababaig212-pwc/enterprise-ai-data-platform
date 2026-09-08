from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.llm_gateway.config.azure_openai_settings import (
    AzureOpenAISettings,
)
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ai_platform.llm_gateway.providers.azure_openai_provider import (
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    AzureOpenAIProvider,
)


@pytest.fixture
def settings():
    return AzureOpenAISettings(
        api_key="test-key",
        endpoint="https://test-resource.openai.azure.com",
        api_version="2024-10-21",
        chat_deployment="chat-deployment",
        embedding_deployment="embedding-deployment",
        timeout=30,
        max_retries=0,
    )


@pytest.fixture
def provider(settings):
    return AzureOpenAIProvider(settings=settings)


@pytest.mark.asyncio
async def test_chat(provider):
    fake_message = MagicMock()
    fake_message.content = "Hello from Azure"

    fake_choice = MagicMock()
    fake_choice.message = fake_message

    fake_usage = MagicMock()
    fake_usage.prompt_tokens = 10
    fake_usage.completion_tokens = 20

    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = fake_usage

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "azure-openai-chat",
        }
    )

    assert result["reply"] == "Hello from Azure"
    assert result["usage"] == {
        "tokens_in": 10,
        "tokens_out": 20,
    }
    assert result["tool_calls"] == []

    fake_client.chat.completions.create.assert_awaited_once_with(
        model="chat-deployment",
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        temperature=0.7,
        max_tokens=1024,
    )


@pytest.mark.asyncio
async def test_chat_structured_messages(provider):
    fake_message = MagicMock()
    fake_message.content = "Structured response"

    fake_choice = MagicMock()
    fake_choice.message = fake_message

    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    messages = [
        {
            "role": "system",
            "content": "You are an enterprise AI assistant.",
        },
        {
            "role": "user",
            "content": "Explain RAG.",
        },
    ]

    result = await provider.chat(
        {
            "prompt": "Explain RAG.",
            "messages": messages,
            "model": "azure-openai-chat",
            "temperature": 0.2,
            "max_tokens": 512,
        }
    )

    assert result["reply"] == "Structured response"

    fake_client.chat.completions.create.assert_awaited_once_with(
        model="chat-deployment",
        messages=messages,
        temperature=0.2,
        max_tokens=512,
    )


@pytest.mark.asyncio
async def test_chat_tool_calls(provider):
    fake_function = MagicMock()
    fake_function.name = "lookup_customer"
    fake_function.arguments = '{"customer_id": "123"}'

    fake_tool_call = MagicMock()
    fake_tool_call.id = "call_123"
    fake_tool_call.function = fake_function

    fake_message = MagicMock()
    fake_message.content = None
    fake_message.tool_calls = [fake_tool_call]

    fake_choice = MagicMock()
    fake_choice.message = fake_message

    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.chat(
        {
            "prompt": "Find customer 123",
            "model": "azure-openai-chat",
            "tools": [
                {
                    "name": "lookup_customer",
                    "description": "Look up a customer.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "customer_id": {
                                "type": "string",
                            }
                        },
                    },
                }
            ],
        }
    )

    assert result["tool_calls"] == [
        AgentToolCall(
            call_id="call_123",
            name="lookup_customer",
            arguments={"customer_id": "123"},
        )
    ]


@pytest.mark.asyncio
async def test_stream(provider):
    class FakeDelta:
        content = None

    class FakeChoice:
        def __init__(self, content):
            self.delta = FakeDelta()
            self.delta.content = content

    class FakeChunk:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    async def fake_stream():
        yield FakeChunk("azure-")
        yield FakeChunk("chunk1")
        yield FakeChunk("azure-")
        yield FakeChunk("chunk2")

    fake_client = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=fake_stream())

    provider.client = fake_client

    chunks = []

    async for chunk in provider.stream(
        {
            "prompt": "Hello",
            "model": "azure-openai-chat",
        }
    ):
        chunks.append(chunk)

    assert chunks == [
        "azure-",
        "chunk1",
        "azure-",
        "chunk2",
    ]

    fake_client.chat.completions.create.assert_awaited_once_with(
        model="chat-deployment",
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        temperature=0.7,
        max_tokens=1024,
        stream=True,
    )


@pytest.mark.asyncio
async def test_embeddings(provider):
    fake_data = MagicMock()
    fake_data.embedding = [0.1, 0.2, 0.3]

    fake_response = MagicMock()
    fake_response.data = [fake_data]

    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.embeddings(
        {
            "model": "azure-openai-embedding",
            "text": "hello world",
        }
    )

    assert result == [0.1, 0.2, 0.3]

    fake_client.embeddings.create.assert_awaited_once_with(
        model="embedding-deployment",
        input="hello world",
    )


@pytest.mark.asyncio
async def test_embeddings_invalid_model(provider):
    with pytest.raises(ValueError):
        await provider.embeddings(
            {
                "model": "bad-model",
                "text": "hello",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_without_text(provider):
    with pytest.raises(
        ValueError,
        match="Embedding input text must not be empty.",
    ):
        await provider.embeddings(
            {
                "model": "azure-openai-embedding",
            }
        )


@pytest.mark.asyncio
async def test_without_configuration():
    settings = AzureOpenAISettings()

    provider = AzureOpenAIProvider(settings=settings)

    assert provider.client is None

    with pytest.raises(ProviderAuthenticationError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "azure-openai-chat",
            }
        )


@pytest.mark.asyncio
async def test_health(provider):
    result = await provider.health_check()

    assert result["status"] == "ok"
    assert result["configured"] is True
    assert result["endpoint"] == ("https://test-resource.openai.azure.com")
    assert result["api_version"] == "2024-10-21"
    assert result["chat_deployment"] == "chat-deployment"
    assert result["embedding_deployment"] == "embedding-deployment"


@pytest.mark.asyncio
async def test_models(provider):
    assert await provider.list_models() == [
        "azure-openai-chat",
        "azure-openai-embedding",
    ]


def test_supported_chat_models(provider):
    assert set(provider.supported_chat_models()) == SUPPORTED_CHAT_MODELS


def test_supported_embedding_models(provider):
    assert set(provider.supported_embedding_models()) == SUPPORTED_EMBEDDING_MODELS


def test_supported_stream_models(provider):
    assert set(provider.supported_stream_models()) == SUPPORTED_CHAT_MODELS


@pytest.mark.asyncio
async def test_chat_authentication_error(provider):
    fake_client = MagicMock()

    fake_client.chat.completions.create.side_effect = AuthenticationError(
        "bad key",
        response=MagicMock(),
        body={},
    )

    provider.client = fake_client

    with pytest.raises(ProviderAuthenticationError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "azure-openai-chat",
            }
        )


@pytest.mark.asyncio
async def test_chat_timeout(provider):
    fake_client = MagicMock()

    fake_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

    provider.client = fake_client

    with pytest.raises(ProviderTimeoutError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "azure-openai-chat",
            }
        )


@pytest.mark.asyncio
async def test_chat_connection_error(provider):
    fake_client = MagicMock()

    fake_client.chat.completions.create.side_effect = APIConnectionError(request=MagicMock())

    provider.client = fake_client

    with pytest.raises(ProviderConnectionError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "azure-openai-chat",
            }
        )


@pytest.mark.asyncio
async def test_chat_rate_limit(provider):
    fake_client = MagicMock()

    fake_client.chat.completions.create.side_effect = RateLimitError(
        "rate limited",
        response=MagicMock(),
        body={
            "error": {
                "code": "rate_limit_exceeded",
                "message": "too many requests",
            }
        },
    )

    provider.client = fake_client

    with pytest.raises(ProviderRateLimitError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "azure-openai-chat",
            }
        )


@pytest.mark.asyncio
async def test_chat_quota(provider):
    fake_client = MagicMock()

    fake_client.chat.completions.create.side_effect = RateLimitError(
        "quota exceeded",
        response=MagicMock(),
        body={
            "error": {
                "code": "insufficient_quota",
                "message": "quota exhausted",
            }
        },
    )

    provider.client = fake_client

    with pytest.raises(ProviderQuotaExceededError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "azure-openai-chat",
            }
        )
