from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


def test_embeddings_success():

    with patch(
        "ai_platform.llm_gateway.providers.openai_provider.OpenAIProvider.embeddings"
    ) as mock_embeddings:

        mock_embeddings.return_value = [0.1, 0.2, 0.3]

        response = client.post(
            "/v1/embeddings",
            headers={
                "x-api-key": "super-secret-key",
            },
            json={
                "provider": "openai",
                "model": "openai-embedding",
                "text": "hello world",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["vector"] == [0.1, 0.2, 0.3]

    assert "metrics" in body
