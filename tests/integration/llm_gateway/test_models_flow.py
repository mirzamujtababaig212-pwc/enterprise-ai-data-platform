from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


def test_models():

    with patch(
        "ai_platform.llm_gateway.providers.openai_provider.OpenAIProvider.list_models"
    ) as mock_models:

        mock_models.return_value = [
            "gpt-4.1",
            "gpt-4o",
        ]

        response = client.get(
            "/v1/models",
            headers={
                "x-api-key": "super-secret-key",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert "openai" in body

    assert "gpt-4.1" in body["openai"]
