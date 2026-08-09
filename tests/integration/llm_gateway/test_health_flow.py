from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app

client = TestClient(app)


def test_health():

    with patch(
        "ai_platform.llm_gateway.providers.openai_provider.OpenAIProvider.health_check"
    ) as mock_health:
        mock_health.return_value = {
            "status": "ok",
            "configured": True,
        }

        response = client.get(
            "/v1/health",
            headers={
                "x-api-key": "super-secret-key",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"

    assert "providers" in body

    assert body["providers"]["openai"]["status"] == "ok"
