from types import SimpleNamespace

import pytest

from ai_platform.llm_gateway.providers.anthropic_provider import (
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    AnthropicProvider,
)


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.create_kwargs = None

    async def create(self, **kwargs):
        self.create_kwargs = kwargs
        return self.response


class FakeStreamContext:
    def __init__(self, chunks):
        self.text_stream = self._stream(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def _stream(self, chunks):
        for chunk in chunks:
            yield chunk


class FakeMessagesWithStream(FakeMessages):
    def __init__(self, response, chunks):
        super().__init__(response)
        self.stream_kwargs = None
        self.chunks = chunks

    def stream(self, **kwargs):
        self.stream_kwargs = kwargs
        return FakeStreamContext(self.chunks)


class FakeClient:
    def __init__(self, response=None, chunks=None):
        self.messages = FakeMessagesWithStream(
            response=response,
            chunks=chunks or [],
        )


def make_response(
    *,
    text="Hello from Claude",
    tool_calls=None,
    input_tokens=12,
    output_tokens=7,
):
    content = []

    if text:
        content.append(
            SimpleNamespace(
                type="text",
                text=text,
            )
        )

    for tool_call in tool_calls or []:
        content.append(
            SimpleNamespace(
                type="tool_use",
                id=tool_call["id"],
                name=tool_call["name"],
                input=tool_call["input"],
            )
        )

    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


@pytest.fixture
def response():
    return make_response()


@pytest.fixture
def fake_client(response):
    return FakeClient(
        response=response,
        chunks=["Hello ", "from ", "Claude"],
    )


@pytest.fixture
def provider(fake_client):
    return AnthropicProvider(client=fake_client)


@pytest.mark.asyncio
async def test_chat_calls_anthropic_messages_api(provider, fake_client):
    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "anthropic-chat",
            "temperature": 0.7,
        }
    )

    assert result["reply"] == "Hello from Claude"
    assert result["usage"] == {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "total_tokens": 19,
    }
    assert result["tool_calls"] == []

    kwargs = fake_client.messages.create_kwargs

    assert "temperature" not in kwargs
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


@pytest.mark.asyncio
async def test_chat_maps_system_messages_and_tools(provider, fake_client):
    result = await provider.chat(
        {
            "model": "anthropic-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an enterprise AI assistant.",
                },
                {
                    "role": "user",
                    "content": "Find the customer.",
                },
            ],
            "max_tokens": 512,
            "tools": [
                {
                    "name": "find_customer",
                    "description": "Find a customer by ID.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                        },
                        "required": ["customer_id"],
                    },
                }
            ],
        }
    )

    assert result["reply"] == "Hello from Claude"

    kwargs = fake_client.messages.create_kwargs

    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 512
    assert kwargs["system"] == "You are an enterprise AI assistant."
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": "Find the customer.",
        }
    ]
    assert kwargs["tools"] == [
        {
            "name": "find_customer",
            "description": "Find a customer by ID.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                },
                "required": ["customer_id"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_chat_extracts_tool_calls():
    response = make_response(
        text="",
        tool_calls=[
            {
                "id": "toolu_123",
                "name": "find_customer",
                "input": {
                    "customer_id": "C123",
                },
            }
        ],
    )
    client = FakeClient(response=response)
    provider = AnthropicProvider(client=client)

    result = await provider.chat(
        {
            "prompt": "Find customer C123.",
            "model": "anthropic-chat",
        }
    )

    assert len(result["tool_calls"]) == 1

    tool_call = result["tool_calls"][0]

    assert tool_call.call_id == "toolu_123"
    assert tool_call.name == "find_customer"
    assert tool_call.arguments == {
        "customer_id": "C123",
    }


@pytest.mark.asyncio
async def test_stream_calls_anthropic_streaming_api(provider, fake_client):
    chunks = []

    async for chunk in provider.stream(
        {
            "prompt": "Hello",
            "model": "anthropic-chat",
        }
    ):
        chunks.append(chunk)

    assert chunks == [
        "Hello ",
        "from ",
        "Claude",
    ]

    kwargs = fake_client.messages.stream_kwargs

    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["max_tokens"] == 1024
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": "Hello",
        }
    ]


@pytest.mark.asyncio
async def test_embeddings_are_explicitly_unsupported(provider):
    with pytest.raises(
        ValueError,
        match="does not provide an embeddings capability",
    ):
        await provider.embeddings(
            {
                "model": "anthropic-embedding",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_invalid_model(provider):
    with pytest.raises(
        ValueError,
        match="does not provide an embeddings capability",
    ):
        await provider.embeddings(
            {
                "model": "bad-model",
            }
        )


@pytest.mark.asyncio
async def test_health_reports_configured(provider):
    assert await provider.health_check() == {
        "status": "configured",
        "provider": "anthropic",
    }


@pytest.mark.asyncio
async def test_unconfigured_provider_reports_unconfigured():
    provider = AnthropicProvider(api_key="")

    assert await provider.health_check() == {
        "status": "unconfigured",
        "provider": "anthropic",
    }


@pytest.mark.asyncio
async def test_models(provider):
    assert await provider.list_models() == [
        "anthropic-chat",
    ]


def test_supported_chat_models(provider):
    assert set(provider.supported_chat_models()) == SUPPORTED_CHAT_MODELS


def test_supported_embedding_models(provider):
    assert set(provider.supported_embedding_models()) == SUPPORTED_EMBEDDING_MODELS


def test_supported_stream_models(provider):
    assert set(provider.supported_stream_models()) == SUPPORTED_CHAT_MODELS
