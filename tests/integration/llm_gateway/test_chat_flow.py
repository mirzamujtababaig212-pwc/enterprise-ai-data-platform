from unittest.mock import AsyncMock, MagicMock


def test_chat_success(client):
    """
    Full integration test.

    Exercises:

        FastAPI
          ↓
        Authentication middleware
          ↓
        Request middleware
          ↓
        Router
          ↓
        Capability Service
          ↓
        ProviderFactory
          ↓
        REAL OpenAIProvider
          ↓
        FAKE async OpenAI client
    """

    from ai_platform.llm_gateway.providers.provider_factory import (
        ProviderFactory,
    )

    provider = ProviderFactory.get_provider("openai")

    assert provider is not None

    fake_response = MagicMock()

    fake_response.output_text = "Hello from OpenAI"

    fake_response.usage.input_tokens = 4
    fake_response.usage.output_tokens = 6

    fake_client = MagicMock()

    fake_client.responses = MagicMock()

    fake_client.responses.create = AsyncMock(return_value=fake_response)

    original_client = provider.client

    try:
        provider.client = fake_client

        response = client.post(
            "/v1/chat",
            headers={
                "x-api-key": "super-secret-key",
            },
            json={
                "provider": "openai",
                "model": "gpt-4.1",
                "prompt": "Hello",
            },
        )

    finally:
        provider.client = original_client

    assert response.status_code == 200

    body = response.json()

    assert body["reply"] == "Hello from OpenAI"

    assert body["metrics"]["tokens_in"] == 4
    assert body["metrics"]["tokens_out"] == 6
    assert body["metrics"]["status"] == "success"

    fake_client.responses.create.assert_awaited_once_with(
        model="gpt-4.1",
        input="Hello",
    )
