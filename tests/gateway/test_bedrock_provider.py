import io
import json
from unittest.mock import Mock

import pytest

from ai_platform.llm_gateway.config.bedrock_settings import BedrockSettings
from ai_platform.llm_gateway.providers.bedrock_provider import (
    SUPPORTED_CHAT_MODELS,
    SUPPORTED_EMBEDDING_MODELS,
    BedrockProvider,
)


@pytest.fixture
def settings():
    return BedrockSettings(
        access_key_id="test-access-key",
        secret_access_key="test-secret-key",
        region="us-east-1",
        chat_model="amazon.nova-micro-v1:0",
        embedding_model="amazon.titan-embed-text-v2:0",
        embedding_dimensions=1024,
        embedding_normalize=True,
    )


@pytest.fixture
def client():
    return Mock()


@pytest.fixture
def provider(client, settings):
    return BedrockProvider(
        client=client,
        settings=settings,
    )


def test_provider_uses_boto3_credential_chain_without_static_credentials(monkeypatch):
    settings = BedrockSettings(
        region="us-east-1",
        chat_model="amazon.nova-micro-v1:0",
        embedding_model="amazon.titan-embed-text-v2:0",
        embedding_dimensions=1024,
        embedding_normalize=True,
    )
    client = Mock()
    boto3_client = Mock(return_value=client)

    monkeypatch.setattr(
        "ai_platform.llm_gateway.providers.bedrock_provider.boto3.client",
        boto3_client,
    )

    provider = BedrockProvider(
        settings=settings,
    )

    assert provider.client is client
    boto3_client.assert_called_once()
    assert boto3_client.call_args.args == ("bedrock-runtime",)
    assert boto3_client.call_args.kwargs["region_name"] == "us-east-1"
    assert "aws_access_key_id" not in boto3_client.call_args.kwargs
    assert "aws_secret_access_key" not in boto3_client.call_args.kwargs


@pytest.mark.asyncio
async def test_chat(provider, client):
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Hello from Bedrock."},
                ],
            }
        },
        "usage": {
            "inputTokens": 7,
            "outputTokens": 4,
        },
        "stopReason": "end_turn",
    }

    result = await provider.chat(
        {
            "prompt": "Hello",
            "model": "bedrock-chat",
        }
    )

    assert result["reply"] == "Hello from Bedrock."
    assert result["usage"] == {
        "tokens_in": 7,
        "tokens_out": 4,
    }
    assert result["stop_reason"] == "end_turn"

    client.converse.assert_called_once()

    kwargs = client.converse.call_args.kwargs

    assert kwargs["modelId"] == "amazon.nova-micro-v1:0"
    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [{"text": "Hello"}],
        }
    ]


@pytest.mark.asyncio
async def test_chat_with_system_message(provider, client):
    client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "Answer"},
                ],
            }
        },
        "usage": {},
    }

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
            "model": "bedrock-chat",
        }
    )

    assert result["reply"] == "Answer"

    kwargs = client.converse.call_args.kwargs

    assert kwargs["system"] == [
        {"text": "You are helpful."},
    ]

    assert kwargs["messages"] == [
        {
            "role": "user",
            "content": [{"text": "Hello"}],
        }
    ]


@pytest.mark.asyncio
async def test_stream(provider, client):
    client.converse_stream.return_value = {
        "stream": iter(
            [
                {
                    "contentBlockDelta": {
                        "delta": {
                            "text": "Hello ",
                        }
                    }
                },
                {
                    "contentBlockDelta": {
                        "delta": {
                            "text": "Bedrock!",
                        }
                    }
                },
            ]
        )
    }

    chunks = []

    async for chunk in provider.stream(
        {
            "prompt": "Hello",
            "model": "bedrock-chat",
        }
    ):
        chunks.append(chunk)

    assert chunks == [
        "Hello ",
        "Bedrock!",
    ]


@pytest.mark.asyncio
async def test_embeddings(provider, client):
    vector = [0.1, 0.2, 0.3]

    client.invoke_model.return_value = {
        "body": io.BytesIO(
            json.dumps(
                {
                    "embedding": vector,
                    "inputTextTokenCount": 3,
                }
            ).encode()
        )
    }

    result = await provider.embeddings(
        {
            "model": "bedrock-embedding",
            "text": "Hello",
        }
    )

    assert result == vector

    client.invoke_model.assert_called_once()

    kwargs = client.invoke_model.call_args.kwargs

    assert kwargs["modelId"] == "amazon.titan-embed-text-v2:0"

    body = json.loads(kwargs["body"])

    assert body == {
        "inputText": "Hello",
        "dimensions": 1024,
        "normalize": True,
    }


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
async def test_embeddings_empty_text(provider):
    with pytest.raises(ValueError):
        await provider.embeddings(
            {
                "model": "bedrock-embedding",
                "text": "",
            }
        )


@pytest.mark.asyncio
async def test_health(provider):
    result = await provider.health_check()

    assert result["status"] == "ok"
    assert result["configured"] is True
    assert result["region"] == "us-east-1"
    assert result["chat_model"] == "amazon.nova-micro-v1:0"
    assert result["embedding_model"] == "amazon.titan-embed-text-v2:0"
    assert result["embedding_dimensions"] == 1024


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


def test_supported_stream_models(provider):
    assert set(provider.supported_stream_models()) == SUPPORTED_CHAT_MODELS
