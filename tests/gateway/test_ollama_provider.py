from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.llm_gateway.config.ollama_settings import OllamaSettings
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderExecutionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ai_platform.llm_gateway.providers.ollama_provider import (
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    OllamaProvider,
)


@pytest.fixture
def settings():
    return OllamaSettings(
        base_url="http://ollama.test:11434",
        chat_model="llama3.2",
        embedding_model="nomic-embed-text",
        timeout=30.0,
    )


@pytest.fixture
def client():
    return MagicMock(spec=httpx.AsyncClient)


@pytest.fixture
def provider(client, settings):
    return OllamaProvider(
        client=client,
        settings=settings,
    )


def make_response(
    status_code: int,
    json_data,
) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request(
            "GET",
            "http://ollama.test:11434",
        ),
    )


@pytest.mark.asyncio
async def test_chat(provider, client):
    client.post = AsyncMock(
        return_value=make_response(
            200,
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello from Ollama",
                },
                "prompt_eval_count": 12,
                "eval_count": 7,
            },
        )
    )

    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "ollama-chat",
        }
    )

    assert result == {
        "reply": "Hello from Ollama",
        "usage": {
            "tokens_in": 12,
            "tokens_out": 7,
        },
        "tool_calls": [],
    }

    client.post.assert_awaited_once()

    _, kwargs = client.post.call_args

    assert kwargs["json"] == {
        "model": "llama3.2",
        "messages": [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        "stream": False,
    }


@pytest.mark.asyncio
async def test_chat_with_messages(provider, client):
    client.post = AsyncMock(
        return_value=make_response(
            200,
            {
                "message": {
                    "role": "assistant",
                    "content": "Response",
                },
            },
        )
    )

    result = await provider.chat(
        {
            "model": "ollama-chat",
            "messages": [
                {
                    "role": "system",
                    "content": "You are helpful.",
                },
                {
                    "role": "user",
                    "content": "Hello",
                },
            ],
            "temperature": 0.2,
        }
    )

    assert result["reply"] == "Response"

    _, kwargs = client.post.call_args

    assert kwargs["json"]["model"] == "llama3.2"
    assert kwargs["json"]["messages"] == [
        {
            "role": "system",
            "content": "You are helpful.",
        },
        {
            "role": "user",
            "content": "Hello",
        },
    ]
    assert kwargs["json"]["options"] == {
        "temperature": 0.2,
    }


@pytest.mark.asyncio
async def test_chat_with_tools(provider, client):
    client.post = AsyncMock(
        return_value=make_response(
            200,
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {
                                    "city": "Hyderabad",
                                },
                            }
                        }
                    ],
                },
            },
        )
    )

    result = await provider.chat(
        {
            "model": "ollama-chat",
            "prompt": "What is the weather?",
            "tools": [
                {
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                }
                            },
                        },
                    }
                }
            ],
        }
    )

    assert result["tool_calls"] == [
        AgentToolCall(
            call_id="ollama-1",
            name="get_weather",
            arguments={
                "city": "Hyderabad",
            },
        )
    ]

    _, kwargs = client.post.call_args

    assert kwargs["json"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                        }
                    },
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_stream(provider, client):
    response = MagicMock()
    response.status_code = 200

    async def lines():
        yield '{"message":{"content":"Hello"}}'
        yield '{"message":{"content":" world"}}'
        yield '{"done":true}'

    response.aiter_lines.return_value = lines()

    stream_context = MagicMock()
    stream_context.__aenter__ = AsyncMock(return_value=response)
    stream_context.__aexit__ = AsyncMock(return_value=None)

    client.stream = MagicMock(return_value=stream_context)

    chunks = []

    async for chunk in provider.stream(
        {
            "prompt": "Hello",
            "model": "ollama-chat",
        }
    ):
        chunks.append(chunk)

    assert chunks == [
        "Hello",
        " world",
    ]

    client.stream.assert_called_once()

    _, kwargs = client.stream.call_args

    assert kwargs["json"] == {
        "model": "llama3.2",
        "messages": [
            {
                "role": "user",
                "content": "Hello",
            }
        ],
        "stream": True,
    }


@pytest.mark.asyncio
async def test_embeddings(provider, client):
    client.post = AsyncMock(
        return_value=make_response(
            200,
            {
                "embeddings": [
                    [
                        0.1,
                        0.2,
                        0.3,
                    ]
                ]
            },
        )
    )

    result = await provider.embeddings(
        {
            "model": "ollama-embedding",
            "input": "Hello",
        }
    )

    assert result == [
        0.1,
        0.2,
        0.3,
    ]

    _, kwargs = client.post.call_args

    assert kwargs["json"] == {
        "model": "nomic-embed-text",
        "input": "Hello",
    }


@pytest.mark.asyncio
async def test_embeddings_supports_text_alias(provider, client):
    client.post = AsyncMock(
        return_value=make_response(
            200,
            {
                "embeddings": [
                    [
                        0.1,
                        0.2,
                    ]
                ]
            },
        )
    )

    result = await provider.embeddings(
        {
            "model": "ollama-embedding",
            "text": "Hello",
        }
    )

    assert result == [
        0.1,
        0.2,
    ]


@pytest.mark.asyncio
async def test_embeddings_invalid_model(provider):
    with pytest.raises(ValueError):
        await provider.embeddings(
            {
                "model": "bad-model",
                "input": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_missing_input(provider):
    with pytest.raises(ValueError):
        await provider.embeddings(
            {
                "model": "ollama-embedding",
            }
        )


@pytest.mark.asyncio
async def test_health(provider, client):
    client.get = AsyncMock(
        return_value=make_response(
            200,
            {
                "models": [
                    {
                        "name": "llama3.2",
                    },
                    {
                        "name": "nomic-embed-text",
                    },
                ]
            },
        )
    )

    result = await provider.health_check()

    assert result == {
        "status": "ok",
        "configured": True,
        "base_url": "http://ollama.test:11434",
        "chat_model": "llama3.2",
        "embedding_model": "nomic-embed-text",
        "models_available": 2,
    }


@pytest.mark.asyncio
async def test_list_models(provider, client):
    client.get = AsyncMock(
        return_value=make_response(
            200,
            {
                "models": [
                    {
                        "name": "llama3.2",
                    },
                    {
                        "name": "nomic-embed-text",
                    },
                ]
            },
        )
    )

    result = await provider.list_models()

    assert result == [
        "llama3.2",
        "nomic-embed-text",
    ]


def test_supported_chat_models(provider):
    assert set(provider.supported_chat_models()) == SUPPORTED_CHAT_MODELS


def test_supported_embedding_models(provider):
    assert set(provider.supported_embedding_models()) == SUPPORTED_EMBEDDING_MODELS


def test_supported_stream_models(provider):
    assert provider.supported_stream_models() == [
        "ollama-chat",
    ]


@pytest.mark.asyncio
async def test_chat_invalid_model(provider):
    with pytest.raises(ValueError):
        await provider.chat(
            {
                "model": "bad-model",
                "prompt": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_chat_missing_prompt(provider):
    with pytest.raises(ValueError):
        await provider.chat(
            {
                "model": "ollama-chat",
            }
        )


@pytest.mark.asyncio
async def test_chat_authentication_error(provider, client):
    client.post = AsyncMock(
        return_value=make_response(
            401,
            {
                "error": "unauthorized",
            },
        )
    )

    with pytest.raises(ProviderAuthenticationError):
        await provider.chat(
            {
                "model": "ollama-chat",
                "prompt": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_chat_rate_limit_error(provider, client):
    client.post = AsyncMock(
        return_value=make_response(
            429,
            {
                "error": "rate limited",
            },
        )
    )

    with pytest.raises(ProviderRateLimitError):
        await provider.chat(
            {
                "model": "ollama-chat",
                "prompt": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_chat_timeout_error(provider, client):
    client.post = AsyncMock(
        side_effect=httpx.ReadTimeout(
            "timed out",
            request=httpx.Request(
                "POST",
                "http://ollama.test:11434/api/chat",
            ),
        )
    )

    with pytest.raises(ProviderTimeoutError):
        await provider.chat(
            {
                "model": "ollama-chat",
                "prompt": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_chat_connection_error(provider, client):
    client.post = AsyncMock(
        side_effect=httpx.ConnectError(
            "connection failed",
            request=httpx.Request(
                "POST",
                "http://ollama.test:11434/api/chat",
            ),
        )
    )

    with pytest.raises(ProviderConnectionError):
        await provider.chat(
            {
                "model": "ollama-chat",
                "prompt": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_chat_server_error(provider, client):
    client.post = AsyncMock(
        return_value=make_response(
            500,
            {
                "error": "server failure",
            },
        )
    )

    with pytest.raises(ProviderExecutionError):
        await provider.chat(
            {
                "model": "ollama-chat",
                "prompt": "Hello",
            }
        )
