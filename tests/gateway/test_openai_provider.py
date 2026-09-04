from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import json
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
from ai_platform.agents.tool_calls import AgentToolCall


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
async def test_embeddings_without_text():

    provider = OpenAIProvider()

    with pytest.raises(
        ValueError,
        match="Embedding input text must not be empty.",
    ):
        await provider.embeddings(
            {
                "model": "openai-embedding",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_with_whitespace_text():

    provider = OpenAIProvider()

    with pytest.raises(
        ValueError,
        match="Embedding input text must not be empty.",
    ):
        await provider.embeddings(
            {
                "model": "openai-embedding",
                "text": "   ",
            }
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
async def test_chat_passes_generation_parameters_to_openai():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.output_text = "Configured response"
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.chat(
        {
            "prompt": "Explain RAG.",
            "model": "gpt-4o",
            "temperature": 0.15,
            "max_tokens": 768,
        }
    )

    assert result["reply"] == "Configured response"

    fake_client.responses.create.assert_awaited_once_with(
        model="gpt-4o",
        input="Explain RAG.",
        temperature=0.15,
        max_output_tokens=768,
    )


@pytest.mark.asyncio
async def test_chat_passes_generation_parameters_with_structured_messages():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.output_text = "Configured structured response"
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

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

    await provider.chat(
        {
            "prompt": "Explain RAG.",
            "messages": messages,
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 512,
        }
    )

    fake_client.responses.create.assert_awaited_once_with(
        model="gpt-4o",
        input=messages,
        temperature=0.2,
        max_output_tokens=512,
    )


@pytest.mark.asyncio
async def test_chat_success_with_structured_messages():

    provider = OpenAIProvider()

    fake_usage = MagicMock()
    fake_usage.input_tokens = 15
    fake_usage.output_tokens = 25

    fake_response = MagicMock()
    fake_response.output_text = "Hello from structured messages"
    fake_response.usage = fake_usage

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

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
            "model": "gpt-4o",
        }
    )

    assert result["reply"] == "Hello from structured messages"
    assert result["usage"]["tokens_in"] == 15
    assert result["usage"]["tokens_out"] == 25

    fake_client.responses.create.assert_awaited_once_with(
        model="gpt-4o",
        input=messages,
        temperature=0.7,
        max_output_tokens=1024,
    )


@pytest.mark.asyncio
async def test_chat_structured_messages_preserve_order():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.output_text = "Ordered response"
    fake_response.usage = None

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    messages = [
        {
            "role": "system",
            "content": "System instruction",
        },
        {
            "role": "user",
            "content": "First user message",
        },
        {
            "role": "assistant",
            "content": "Previous assistant response",
        },
        {
            "role": "user",
            "content": "Current user message",
        },
    ]

    await provider.chat(
        {
            "prompt": "Current user message",
            "messages": messages,
            "model": "gpt-4o",
        }
    )

    fake_client.responses.create.assert_awaited_once_with(
        model="gpt-4o",
        input=messages,
        temperature=0.7,
        max_output_tokens=1024,
    )


@pytest.mark.asyncio
async def test_chat_rejects_empty_structured_messages():

    provider = OpenAIProvider()

    provider.client = MagicMock()

    with pytest.raises(
        ValueError,
        match="OpenAI chat messages must not be empty.",
    ):
        await provider.chat(
            {
                "prompt": "Hello",
                "messages": [],
                "model": "gpt-4o",
            }
        )


@pytest.mark.asyncio
async def test_chat_rejects_invalid_structured_message():

    provider = OpenAIProvider()

    provider.client = MagicMock()

    with pytest.raises(
        ValueError,
        match="OpenAI chat message content must not be empty.",
    ):
        await provider.chat(
            {
                "prompt": "Hello",
                "messages": [
                    {
                        "role": "user",
                        "content": "",
                    }
                ],
                "model": "gpt-4o",
            }
        )


@pytest.mark.asyncio
async def test_chat_rejects_non_list_messages():

    provider = OpenAIProvider()

    provider.client = MagicMock()

    with pytest.raises(
        ValueError,
        match="OpenAI chat messages must be a list.",
    ):
        await provider.chat(
            {
                "prompt": "Hello",
                "messages": "invalid",
                "model": "gpt-4o",
            }
        )


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


@pytest.mark.asyncio
async def test_embeddings_empty_response():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.data = []

    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    with pytest.raises(
        ValueError,
        match="OpenAI embedding response contained no data.",
    ):
        await provider.embeddings(
            {
                "model": "openai-embedding",
                "text": "hello world",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_empty_vector():

    provider = OpenAIProvider()

    fake_data = MagicMock()
    fake_data.embedding = []

    fake_response = MagicMock()
    fake_response.data = [fake_data]

    fake_client = MagicMock()
    fake_client.embeddings.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    with pytest.raises(
        ValueError,
        match="OpenAI embedding response contained an empty vector.",
    ):
        await provider.embeddings(
            {
                "model": "openai-embedding",
                "text": "hello world",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_authentication_error():

    provider = OpenAIProvider()

    fake_client = MagicMock()

    fake_client.embeddings.create.side_effect = AuthenticationError(
        "bad key",
        response=MagicMock(),
        body={},
    )

    provider.client = fake_client

    with pytest.raises(ProviderAuthenticationError):
        await provider.embeddings(
            {
                "model": "openai-embedding",
                "text": "hello world",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_timeout():

    provider = OpenAIProvider()

    fake_client = MagicMock()

    fake_client.embeddings.create.side_effect = APITimeoutError(request=MagicMock())

    provider.client = fake_client

    with pytest.raises(ProviderTimeoutError):
        await provider.embeddings(
            {
                "model": "openai-embedding",
                "text": "hello world",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_connection_error():

    provider = OpenAIProvider()

    fake_client = MagicMock()

    fake_client.embeddings.create.side_effect = APIConnectionError(
        message="connection failed",
        request=MagicMock(),
    )

    provider.client = fake_client

    with pytest.raises(ProviderConnectionError):
        await provider.embeddings(
            {
                "model": "openai-embedding",
                "text": "hello world",
            }
        )


@pytest.mark.asyncio
async def test_chat_success_without_tool_calls_returns_empty_tool_calls():

    provider = OpenAIProvider()

    fake_usage = MagicMock()
    fake_usage.input_tokens = 11
    fake_usage.output_tokens = 22

    fake_response = MagicMock()
    fake_response.output_text = "Hello from OpenAI"
    fake_response.usage = fake_usage
    fake_response.output = []

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
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_chat_extracts_openai_function_call():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.output_text = ""
    fake_response.usage = None

    function_call = MagicMock()
    function_call.type = "function_call"
    function_call.call_id = "call_123"
    function_call.name = "search_documents"
    function_call.arguments = json.dumps(
        {
            "query": "enterprise RAG",
            "top_k": 5,
        }
    )

    fake_response.output = [function_call]

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.chat(
        {
            "prompt": "Search the documents for enterprise RAG.",
            "model": "gpt-4o",
        }
    )

    assert len(result["tool_calls"]) == 1

    tool_call = result["tool_calls"][0]

    assert isinstance(tool_call, AgentToolCall)
    assert tool_call.call_id == "call_123"
    assert tool_call.name == "search_documents"
    assert tool_call.arguments == {
        "query": "enterprise RAG",
        "top_k": 5,
    }


@pytest.mark.asyncio
async def test_chat_extracts_multiple_openai_function_calls():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.output_text = ""
    fake_response.usage = None

    first_call = MagicMock()
    first_call.type = "function_call"
    first_call.call_id = "call_1"
    first_call.name = "search_documents"
    first_call.arguments = '{"query": "RAG"}'

    second_call = MagicMock()
    second_call.type = "function_call"
    second_call.call_id = "call_2"
    second_call.name = "get_document"
    second_call.arguments = '{"document_id": "doc_123"}'

    fake_response.output = [
        first_call,
        second_call,
    ]

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    result = await provider.chat(
        {
            "prompt": "Find the RAG document and retrieve it.",
            "model": "gpt-4o",
        }
    )

    assert len(result["tool_calls"]) == 2

    assert result["tool_calls"][0].call_id == "call_1"
    assert result["tool_calls"][0].name == "search_documents"
    assert result["tool_calls"][0].arguments == {
        "query": "RAG",
    }

    assert result["tool_calls"][1].call_id == "call_2"
    assert result["tool_calls"][1].name == "get_document"
    assert result["tool_calls"][1].arguments == {
        "document_id": "doc_123",
    }


@pytest.mark.asyncio
async def test_chat_rejects_invalid_tool_call_arguments():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.output_text = ""
    fake_response.usage = None

    function_call = MagicMock()
    function_call.type = "function_call"
    function_call.call_id = "call_bad"
    function_call.name = "search_documents"
    function_call.arguments = "{invalid-json"

    fake_response.output = [function_call]

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    with pytest.raises(
        ValueError,
        match="function call arguments were not valid JSON",
    ):
        await provider.chat(
            {
                "prompt": "Search documents.",
                "model": "gpt-4o",
            }
        )


@pytest.mark.asyncio
async def test_chat_rejects_non_object_tool_call_arguments():

    provider = OpenAIProvider()

    fake_response = MagicMock()
    fake_response.output_text = ""
    fake_response.usage = None

    function_call = MagicMock()
    function_call.type = "function_call"
    function_call.call_id = "call_array"
    function_call.name = "search_documents"
    function_call.arguments = '["RAG", "AI"]'

    fake_response.output = [function_call]

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=fake_response)

    provider.client = fake_client

    with pytest.raises(
        ValueError,
        match="arguments must decode to an object",
    ):
        await provider.chat(
            {
                "prompt": "Search documents.",
                "model": "gpt-4o",
            }
        )


@pytest.mark.asyncio
async def test_chat_translates_tool_definitions() -> None:
    provider = OpenAIProvider()

    response = MagicMock()
    response.output_text = "Tool-enabled response"
    response.usage = None
    response.output = []

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(
        return_value=response,
    )

    provider.client = fake_client
    provider.client.responses.create = AsyncMock(
        return_value=response,
    )

    await provider.chat(
        {
            "model": "gpt-4o",
            "prompt": "Find information.",
            "tools": [
                {
                    "name": "search",
                    "description": "Search enterprise knowledge.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                            },
                        },
                        "required": ["query"],
                    },
                },
            ],
        }
    )

    call = provider.client.responses.create.call_args

    assert call.kwargs["tools"] == [
        {
            "type": "function",
            "name": "search",
            "description": "Search enterprise knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                },
                "required": ["query"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_chat_preserves_tool_order() -> None:
    provider = OpenAIProvider()

    response = MagicMock()
    response.output_text = "Tool-enabled response"
    response.usage = None
    response.output = []

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(
        return_value=response,
    )

    provider.client = fake_client
    provider.client.responses.create = AsyncMock(
        return_value=response,
    )

    await provider.chat(
        {
            "model": "gpt-4o",
            "prompt": "Use tools.",
            "tools": [
                {
                    "name": "search",
                    "description": "Search.",
                    "input_schema": {},
                },
                {
                    "name": "calculator",
                    "description": "Calculate.",
                    "input_schema": {},
                },
            ],
        }
    )

    tools = provider.client.responses.create.call_args.kwargs["tools"]

    assert [tool["name"] for tool in tools] == [
        "search",
        "calculator",
    ]


@pytest.mark.asyncio
async def test_chat_omits_tools_when_not_provided() -> None:
    provider = OpenAIProvider()

    response = MagicMock()
    response.output_text = "Tool-enabled response"
    response.usage = None
    response.output = []

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(
        return_value=response,
    )

    provider.client = fake_client
    provider.client.responses.create = AsyncMock(
        return_value=response,
    )

    await provider.chat(
        {
            "model": "gpt-4o",
            "prompt": "Hello.",
        }
    )

    assert "tools" not in provider.client.responses.create.call_args.kwargs


@pytest.mark.asyncio
async def test_chat_passes_empty_tools_list() -> None:
    provider = OpenAIProvider()

    response = MagicMock()
    response.output_text = "Tool-enabled response"
    response.usage = None
    response.output = []

    fake_client = MagicMock()
    fake_client.responses.create = AsyncMock(
        return_value=response,
    )

    provider.client = fake_client
    provider.client.responses.create = AsyncMock(
        return_value=response,
    )

    await provider.chat(
        {
            "model": "gpt-4o",
            "prompt": "Hello.",
            "tools": [],
        }
    )

    assert provider.client.responses.create.call_args.kwargs["tools"] == []
