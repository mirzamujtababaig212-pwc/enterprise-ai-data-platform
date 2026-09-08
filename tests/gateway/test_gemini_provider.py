from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from ai_platform.agents.tool_calls import AgentToolCall

from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderExecutionError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ai_platform.llm_gateway.config.gemini_settings import GeminiSettings
from ai_platform.llm_gateway.providers.gemini_provider import (
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    GeminiProvider,
)


class FakeModels:
    def __init__(self):
        self.generate_content = AsyncMock()
        self.generate_content_stream = AsyncMock()
        self.embed_content = AsyncMock()


class FakeAio:
    def __init__(self, models):
        self.models = models


class FakeClient:
    def __init__(self):
        self.models = FakeModels()
        self.aio = FakeAio(self.models)


@pytest.fixture
def settings():
    return GeminiSettings(
        api_key="test-key",
        model="gemini-2.5-flash",
        embedding_model="gemini-embedding-001",
        timeout=60.0,
    )


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def provider(client, settings):
    return GeminiProvider(
        client=client,
        settings=settings,
    )


@pytest.mark.asyncio
async def test_chat(provider, client):
    client.models.generate_content.return_value = SimpleNamespace(
        text="Hello from Gemini",
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=7,
        ),
        candidates=[],
    )

    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "gemini-chat",
        }
    )

    assert result == {
        "reply": "Hello from Gemini",
        "usage": {
            "tokens_in": 12,
            "tokens_out": 7,
        },
        "tool_calls": [],
    }

    client.models.generate_content.assert_awaited_once()

    call = client.models.generate_content.await_args

    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert call.kwargs["contents"] == "Hello"


@pytest.mark.asyncio
async def test_chat_supports_messages(provider, client):
    client.models.generate_content.return_value = SimpleNamespace(
        text="Message response",
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=5,
        ),
        candidates=[],
    )

    result = await provider.chat(
        {
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
            "model": "gemini-chat",
        }
    )

    assert result["reply"] == "Message response"

    call = client.models.generate_content.await_args

    assert call.kwargs["model"] == "gemini-2.5-flash"

    config = call.kwargs["config"]

    assert config is not None
    assert config.system_instruction == "You are helpful."


@pytest.mark.asyncio
async def test_stream(provider, client):
    async def fake_stream():
        yield SimpleNamespace(text="Hello ")
        yield SimpleNamespace(text="from Gemini")

    client.models.generate_content_stream.return_value = fake_stream()

    chunks = []

    async for chunk in provider.stream(
        {
            "prompt": "Hello",
            "model": "gemini-chat",
        }
    ):
        chunks.append(chunk)

    assert chunks == [
        "Hello ",
        "from Gemini",
    ]

    client.models.generate_content_stream.assert_awaited_once()

    call = client.models.generate_content_stream.await_args

    assert call.kwargs["model"] == "gemini-2.5-flash"
    assert call.kwargs["contents"] == "Hello"


@pytest.mark.asyncio
async def test_embeddings(provider, client):
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[
            SimpleNamespace(
                values=[0.1, 0.2, 0.3, 0.4],
            )
        ]
    )

    result = await provider.embeddings(
        {
            "model": "gemini-embedding",
            "text": "Enterprise AI",
        }
    )

    assert result == [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    client.models.embed_content.assert_awaited_once()

    call = client.models.embed_content.await_args

    assert call.kwargs["model"] == "gemini-embedding-001"
    assert call.kwargs["contents"] == "Enterprise AI"


@pytest.mark.asyncio
async def test_chat_requires_configuration(settings):
    provider = GeminiProvider(
        client=None,
        settings=GeminiSettings(
            api_key="",
            model=settings.model,
            embedding_model=settings.embedding_model,
            timeout=settings.timeout,
        ),
    )

    with pytest.raises(ProviderAuthenticationError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "gemini-chat",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_invalid_model(provider):
    with pytest.raises(ValueError):
        await provider.embeddings(
            {
                "model": "bad-model",
                "text": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_embeddings_requires_text(provider):
    with pytest.raises(ValueError):
        await provider.embeddings(
            {
                "model": "gemini-embedding",
            }
        )


@pytest.mark.asyncio
async def test_chat_invalid_model(provider):
    with pytest.raises(ValueError):
        await provider.chat(
            {
                "prompt": "Hello",
                "model": "bad-model",
            }
        )


@pytest.mark.asyncio
async def test_health(provider):
    result = await provider.health_check()

    assert result == {
        "status": "ok",
        "configured": True,
        "default_model": "gemini-chat",
        "embedding_model": "gemini-embedding-001",
    }


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


def test_supported_stream_models(provider):
    assert set(provider.supported_stream_models()) == SUPPORTED_CHAT_MODELS


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403])
async def test_chat_maps_authentication_errors(
    provider,
    client,
    status_code,
):
    client.models.generate_content.side_effect = genai_errors.APIError(
        status_code,
        {"error": {"status": "PERMISSION_DENIED", "message": "Unauthorized"}},
    )

    with pytest.raises(ProviderAuthenticationError):
        await provider.chat({"prompt": "Hello", "model": "gemini-chat"})


@pytest.mark.asyncio
async def test_chat_maps_quota_exceeded_error(provider, client):
    client.models.generate_content.side_effect = genai_errors.APIError(
        429,
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "Quota exceeded for this project",
            }
        },
    )

    with pytest.raises(ProviderQuotaExceededError):
        await provider.chat({"prompt": "Hello", "model": "gemini-chat"})


@pytest.mark.asyncio
async def test_chat_maps_rate_limit_error(provider, client):
    client.models.generate_content.side_effect = genai_errors.APIError(
        429,
        {
            "error": {
                "status": "RESOURCE_EXHAUSTED",
                "message": "Too many requests",
            }
        },
    )

    with pytest.raises(ProviderRateLimitError):
        await provider.chat({"prompt": "Hello", "model": "gemini-chat"})


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 504])
async def test_chat_maps_timeout_errors(
    provider,
    client,
    status_code,
):
    client.models.generate_content.side_effect = genai_errors.APIError(
        status_code,
        {"error": {"status": "TIMEOUT", "message": "Request timed out"}},
    )

    with pytest.raises(ProviderTimeoutError):
        await provider.chat({"prompt": "Hello", "model": "gemini-chat"})


@pytest.mark.asyncio
async def test_chat_maps_server_errors(provider, client):
    client.models.generate_content.side_effect = genai_errors.APIError(
        500,
        {"error": {"status": "INTERNAL", "message": "Internal server error"}},
    )

    with pytest.raises(ProviderConnectionError):
        await provider.chat({"prompt": "Hello", "model": "gemini-chat"})


@pytest.mark.asyncio
async def test_chat_maps_other_api_errors(provider, client):
    client.models.generate_content.side_effect = genai_errors.APIError(
        400,
        {"error": {"status": "INVALID_ARGUMENT", "message": "Bad request"}},
    )

    with pytest.raises(ProviderExecutionError):
        await provider.chat({"prompt": "Hello", "model": "gemini-chat"})


@pytest.mark.asyncio
async def test_chat_maps_timeout_exception(provider, client):
    client.models.generate_content.side_effect = TimeoutError("Gemini request timed out")

    with pytest.raises(ProviderTimeoutError):
        await provider.chat({"prompt": "Hello", "model": "gemini-chat"})


@pytest.mark.asyncio
async def test_chat_maps_os_error(provider, client):
    client.models.generate_content.side_effect = OSError("Connection failed")

    with pytest.raises(ProviderConnectionError):
        await provider.chat({"prompt": "Hello", "model": "gemini-chat"})


@pytest.mark.asyncio
async def test_chat_extracts_function_calls(provider, client):
    function_call = genai_types.FunctionCall(
        name="search_knowledge",
        args={
            "query": "enterprise AI architecture",
            "top_k": 5,
        },
    )

    response = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(function_call=function_call),
                    ],
                )
            )
        ],
    )

    client.models.generate_content.return_value = response

    result = await provider.chat(
        {
            "prompt": "Search the knowledge base.",
            "model": "gemini-chat",
            "tools": [
                {
                    "name": "search_knowledge",
                    "description": "Search enterprise knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                }
            ],
        }
    )

    assert len(result["tool_calls"]) == 1

    tool_call = result["tool_calls"][0]

    assert isinstance(tool_call, AgentToolCall)
    assert tool_call.name == "search_knowledge"
    assert tool_call.arguments == {
        "query": "enterprise AI architecture",
        "top_k": 5,
    }
    assert tool_call.call_id.startswith("gemini-")


@pytest.mark.asyncio
async def test_chat_extracts_multiple_function_calls(provider, client):
    response = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name="search_knowledge",
                                args={"query": "Delta Lake"},
                            )
                        ),
                        genai_types.Part(
                            function_call=genai_types.FunctionCall(
                                name="get_document",
                                args={"document_id": "doc-123"},
                            )
                        ),
                    ],
                )
            )
        ],
    )

    client.models.generate_content.return_value = response

    result = await provider.chat(
        {
            "prompt": "Search and retrieve the document.",
            "model": "gemini-chat",
            "tools": [
                {
                    "name": "search_knowledge",
                    "description": "Search enterprise knowledge.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                },
                {
                    "name": "get_document",
                    "description": "Retrieve a document.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "document_id": {"type": "string"},
                        },
                    },
                },
            ],
        }
    )

    assert len(result["tool_calls"]) == 2

    first, second = result["tool_calls"]

    assert isinstance(first, AgentToolCall)
    assert first.name == "search_knowledge"
    assert first.arguments == {"query": "Delta Lake"}
    assert first.call_id.startswith("gemini-")

    assert isinstance(second, AgentToolCall)
    assert second.name == "get_document"
    assert second.arguments == {"document_id": "doc-123"}
    assert second.call_id.startswith("gemini-")

    assert first.call_id != second.call_id
