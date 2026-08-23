from types import SimpleNamespace

import pytest

from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
)
from ai_platform.llm_gateway.providers.openai_provider import OpenAIProvider


def test_provider_initializes_without_credentials(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )
    monkeypatch.delenv(
        "PROVIDER_CREDENTIALS",
        raising=False,
    )

    provider = OpenAIProvider()

    assert provider.default_model == "gpt-4.1-mini"
    assert provider.client is None


def test_provider_initializes_with_credentials(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )

    provider = OpenAIProvider()

    assert provider.settings.api_key == "test-key"
    assert provider.client is not None


def test_supported_chat_models():
    provider = OpenAIProvider(
        settings=type(
            "TestSettings",
            (),
            {
                "api_key": "",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "timeout": 60.0,
                "max_retries": 2,
            },
        )()
    )

    models = provider.supported_chat_models()

    assert "gpt-4.1" in models
    assert "gpt-4.1-mini" in models
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
    assert "o4-mini" in models


def test_supported_embedding_models():
    provider = OpenAIProvider(
        settings=type(
            "TestSettings",
            (),
            {
                "api_key": "",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "timeout": 60.0,
                "max_retries": 2,
            },
        )()
    )

    models = provider.supported_embedding_models()

    assert "openai-embedding" in models


@pytest.mark.asyncio
async def test_chat_without_credentials_fails_cleanly(monkeypatch):
    monkeypatch.delenv(
        "OPENAI_API_KEY",
        raising=False,
    )
    monkeypatch.delenv(
        "PROVIDER_CREDENTIALS",
        raising=False,
    )

    provider = OpenAIProvider()

    with pytest.raises(ProviderAuthenticationError):
        await provider.chat(
            {
                "prompt": "Hello",
            }
        )


@pytest.mark.asyncio
async def test_chat_success():
    response = SimpleNamespace(
        output_text="Hello from OpenAI",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
        ),
    )

    class FakeResponses:
        async def create(self, **kwargs):
            assert kwargs["model"] == "gpt-4.1-mini"
            assert kwargs["input"] == "Hello"
            return response

    class FakeClient:
        responses = FakeResponses()

    provider = OpenAIProvider(
        client=FakeClient(),
        settings=type(
            "TestSettings",
            (),
            {
                "api_key": "test-key",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "timeout": 60.0,
                "max_retries": 2,
            },
        )(),
    )

    result = await provider.chat(
        {
            "prompt": "Hello",
        }
    )

    assert result["reply"] == "Hello from OpenAI"
    assert result["usage"]["tokens_in"] == 10
    assert result["usage"]["tokens_out"] == 5


@pytest.mark.asyncio
async def test_stream_returns_text_deltas():
    events = [
        SimpleNamespace(
            type="response.output_text.delta",
            delta="Hello ",
        ),
        SimpleNamespace(
            type="response.output_text.delta",
            delta="world",
        ),
        SimpleNamespace(
            type="response.completed",
        ),
    ]

    class FakeResponses:
        async def create(self, **kwargs):
            assert kwargs["stream"] is True

            async def generator():
                for event in events:
                    yield event

            return generator()

    class FakeClient:
        responses = FakeResponses()

    provider = OpenAIProvider(
        client=FakeClient(),
        settings=type(
            "TestSettings",
            (),
            {
                "api_key": "test-key",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "timeout": 60.0,
                "max_retries": 2,
            },
        )(),
    )

    chunks = []

    async for chunk in provider.stream(
        {
            "prompt": "Hello",
        }
    ):
        chunks.append(chunk)

    assert chunks == ["Hello ", "world"]


@pytest.mark.asyncio
async def test_embeddings_success():
    response = SimpleNamespace(
        data=[
            SimpleNamespace(
                embedding=[0.1, 0.2, 0.3],
            )
        ]
    )

    class FakeEmbeddings:
        async def create(self, **kwargs):
            assert kwargs["model"] == "text-embedding-3-small"
            assert kwargs["input"] == "Hello"
            return response

    class FakeClient:
        embeddings = FakeEmbeddings()

    provider = OpenAIProvider(
        client=FakeClient(),
        settings=type(
            "TestSettings",
            (),
            {
                "api_key": "test-key",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.openai.com/v1",
                "timeout": 60.0,
                "max_retries": 2,
            },
        )(),
    )

    result = await provider.embeddings(
        {
            "model": "openai-embedding",
            "text": "Hello",
        }
    )

    assert result == [0.1, 0.2, 0.3]
