import json
from types import SimpleNamespace

import pytest
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    ProviderNotFound,
)

from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderConnectionError,
)

from ai_platform.llm_gateway.exceptions.handlers import (
    provider_not_found_handler,
    http_exception_handler,
    validation_exception_handler,
    provider_auth_handler,
    provider_rate_limit_handler,
    provider_timeout_handler,
    provider_connection_handler,
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
        ProviderNotFound("Unknown provider"),
    )

    assert response.status_code == 400

    body = json.loads(response.body)

    assert body["error"] == "Unknown provider"


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
        ProviderAuthenticationError("Invalid API Key"),
    )

    assert response.status_code == 401

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["error"]["code"] == 401
    assert body["error"]["message"] == "Invalid API Key"
    assert body["request_id"] == "req-123"


###############################################################################
# Rate Limit
###############################################################################


@pytest.mark.asyncio
async def test_provider_rate_limit_handler():

    request = DummyRequest()

    response = await provider_rate_limit_handler(
        request,
        ProviderRateLimitError("Rate limit exceeded"),
    )

    assert response.status_code == 429

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["error"]["code"] == 429
    assert body["error"]["message"] == "Rate limit exceeded"
    assert body["request_id"] == "req-123"


###############################################################################
# Timeout
###############################################################################


@pytest.mark.asyncio
async def test_provider_timeout_handler():

    request = DummyRequest()

    response = await provider_timeout_handler(
        request,
        ProviderTimeoutError("Timeout"),
    )

    assert response.status_code == 504

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["error"]["code"] == 504
    assert body["error"]["message"] == "Timeout"
    assert body["request_id"] == "req-123"


###############################################################################
# Connection Error
###############################################################################


@pytest.mark.asyncio
async def test_provider_connection_handler():

    request = DummyRequest()

    response = await provider_connection_handler(
        request,
        ProviderConnectionError("Connection failed"),
    )

    assert response.status_code == 503

    body = json.loads(response.body)

    assert body["status"] == "error"
    assert body["error"]["code"] == 503
    assert body["error"]["message"] == "Connection failed"
    assert body["request_id"] == "req-123"
