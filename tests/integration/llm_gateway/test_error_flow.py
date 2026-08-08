from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


def test_provider_timeout():

    with patch(
        "ai_platform.llm_gateway.providers.openai_provider.OpenAIProvider.chat"
    ) as mock_chat:

        from ai_platform.llm_gateway.exceptions.provider_exceptions import (
            ProviderTimeoutError,
        )

        mock_chat.side_effect = ProviderTimeoutError("timeout")

        response = client.post(
            "/v1/chat",
            headers={
                "x-api-key": "super-secret-key",
            },
            json={
                "provider": "openai",
                "model": "gpt-4.1",
                "prompt": "hello",
            },
        )

    assert response.status_code == 504
