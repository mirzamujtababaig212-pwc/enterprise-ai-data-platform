import json
from types import SimpleNamespace

import pytest
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)
from ai_platform.llm_gateway.exceptions.handlers import (
    http_exception_handler,
    provider_auth_handler,
    provider_connection_handler,
    provider_not_found_handler,
    provider_quota_exceeded_exception_handler,
    provider_rate_limit_handler,
    provider_timeout_handler,
    validation_exception_handler,
)
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderQuotaExceededError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)

###############################################################################
# Helper
###############################################################################


class DummyRequest:
    def __init__(self):
        self.state = SimpleNamespace(request_id="req-123")


###############################################################################
# Provider Not Found
###############################################################################


@pytest.mark.asyncio
async def test_provider_not_found_handler():

    request = DummyRequest()

    response = await provider_not_found_handler(
        request,
        ProviderNotFound("Unknown provider SECRET_MARKER"),
    )

    assert response.status_code == 400

    body = json.loads(response.body)

    assert body["error"] == "Unknown provider or unsupported model."


###############################################################################
# HTTP Exception
###############################################################################


@pytest.mark.asyncio
async def test_http_exception_handler():

    request = DummyRequest()

    response = await http_exception_handler(
        request,
        StarletteHTTPException(
            status_code=404,
            detail="Resource not found",
        ),
    )

    assert response.status_code == 404

    body = json.loads(response.body)

    assert body["error"]["code"] == 404
    assert body["error"]["message"] == "Resource not found"
    assert body["request_id"] == "req-123"


@pytest.mark.asyncio
async def test_http_exception_handler_sanitizes_server_errors():
    request = DummyRequest()

    response = await http_exception_handler(
        request,
        StarletteHTTPException(
            status_code=500,
            detail="SECRET_MARKER internal credentials leaked",
        ),
    )

    assert response.status_code == 500
    body = json.loads(response.body)
    assert body["error"]["code"] == 500
    assert body["error"]["message"] == "Internal server error."
    assert "SECRET_MARKER" not in response.body.decode()
    assert body["request_id"] == "req-123"


###############################################################################
# Validation Exception
###############################################################################


@pytest.mark.asyncio
async def test_validation_exception_handler():

    request = DummyRequest()

    exc = RequestValidationError([])

    response = await validation_exception_handler(
        request,
        exc,
    )

    assert response.status_code == 422

    body = json.loads(response.body)

    assert body["error"]["code"] == 422
    assert body["error"]["message"] == "Request validation failed"
    assert body["request_id"] == "req-123"


###############################################################################
# Authentication Exception
###############################################################################


@pytest.mark.asyncio
async def test_provider_auth_handler():

    request = DummyRequest()

    response = await provider_auth_handler(
        request,
        ProviderAuthenticationError("Invalid API Key SECRET_MARKER"),
    )

    assert response.status_code == 401

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["error"]["code"] == 401
    assert body["error"]["message"] == "Provider authentication failed."
    assert body["request_id"] == "req-123"


###############################################################################
# Rate Limit
###############################################################################


@pytest.mark.asyncio
async def test_provider_rate_limit_handler():

    request = DummyRequest()

    response = await provider_rate_limit_handler(
        request,
        ProviderRateLimitError("Rate limit exceeded SECRET_MARKER"),
    )

    assert response.status_code == 429

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["error"]["code"] == 429
    assert body["error"]["message"] == "Provider rate limit exceeded."
    assert body["request_id"] == "req-123"


###############################################################################
# Timeout
###############################################################################


@pytest.mark.asyncio
async def test_provider_timeout_handler():

    request = DummyRequest()

    response = await provider_timeout_handler(
        request,
        ProviderTimeoutError("Timeout SECRET_MARKER"),
    )

    assert response.status_code == 504

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["error"]["code"] == 504
    assert body["error"]["message"] == "Provider request timed out."
    assert body["request_id"] == "req-123"


###############################################################################
# Connection Error
###############################################################################


@pytest.mark.asyncio
async def test_provider_connection_handler():

    request = DummyRequest()

    response = await provider_connection_handler(
        request,
        ProviderConnectionError("Connection failed SECRET_MARKER"),
    )

    assert response.status_code == 503

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["error"]["code"] == 503
    assert body["error"]["message"] == "Provider connection failed."
    assert body["request_id"] == "req-123"


###############################################################################
# Quota Exceeded
###############################################################################


@pytest.mark.asyncio
async def test_provider_quota_exceeded_handler_sanitizes_message():
    request = DummyRequest()

    response = await provider_quota_exceeded_exception_handler(
        request,
        ProviderQuotaExceededError("Quota failed SECRET_MARKER"),
    )

    assert response.status_code == 429
    body = json.loads(response.body)
    assert body["status"] == "error"
    assert body["error"]["code"] == 429
    assert body["error"]["message"] == "Provider quota exceeded."
    assert "SECRET_MARKER" not in response.body.decode()
    assert body["request_id"] == "req-123"
