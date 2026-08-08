from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from ai_platform.llm_gateway.api.main import app


@pytest.fixture(scope="session")
def client():
    """
    Real FastAPI application.

    Exercises:

    API
      ↓
    Middleware
      ↓
    Router
      ↓
    Capability Service
      ↓
    Provider Factory
      ↓
    Provider
    """
    return TestClient(app)


@pytest.fixture
def api_headers():
    """
    Default headers used by all integration tests.
    """
    return {
        "x-api-key": "test-api-key",
        "Content-Type": "application/json",
    }


@pytest.fixture
def mock_openai_response():
    """
    Mock the external OpenAI SDK while leaving the gateway untouched.
    """
    with patch("ai_platform.llm_gateway.providers.openai_provider.OpenAI") as mock_client:
        yield mock_client
