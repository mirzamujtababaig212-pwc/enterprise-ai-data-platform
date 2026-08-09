import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import Response

from ai_platform.llm_gateway.middleware.request_logging import (
    RequestLoggingMiddleware,
)


@pytest.fixture
def middleware():
    return RequestLoggingMiddleware(MagicMock())


def make_request(
    body=b"",
    headers=None,
    metrics=None,
):
    request = MagicMock(spec=Request)

    request.body = AsyncMock(return_value=body)

    request.method = "POST"

    request.url.path = "/v1/chat"

    request.client.host = "127.0.0.1"

    request.headers = headers or {
        "user-agent": "pytest",
    }

    request.state = MagicMock()

    request.state.request_id = "req-123"

    request.state.metrics = metrics or {}

    return request


@patch(
    "ai_platform.llm_gateway.middleware.request_logging.logger",
)
@pytest.mark.asyncio
async def test_invalid_json_body(
    mock_logger,
    middleware,
):

    request = make_request(
        body=b'{"bad-json"',
    )

    response = Response(status_code=200)

    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(
        request,
        call_next,
    )

    assert result.status_code == 200


@patch(
    "ai_platform.llm_gateway.middleware.request_logging.logger",
)
@pytest.mark.asyncio
async def test_receive_function(
    mock_logger,
    middleware,
):

    payload = {
        "provider": "openai",
        "model": "gpt-4o",
    }

    request = make_request(
        body=json.dumps(payload).encode(),
    )

    response = Response(status_code=200)

    call_next = AsyncMock(return_value=response)

    await middleware.dispatch(
        request,
        call_next,
    )

    request.body.assert_awaited_once()

    mock_logger.info.assert_any_call("Request logging middleware executed")


@patch(
    "ai_platform.llm_gateway.middleware.request_logging.ERRORS_TOTAL",
)
@patch(
    "ai_platform.llm_gateway.middleware.request_logging.logger",
)
@pytest.mark.asyncio
async def test_error_counter(
    mock_logger,
    mock_errors,
    middleware,
):

    request = make_request()

    response = Response(status_code=500)

    call_next = AsyncMock(return_value=response)

    await middleware.dispatch(
        request,
        call_next,
    )

    mock_errors.labels.assert_called_once_with(
        status_code="500",
    )

    mock_errors.labels.return_value.inc.assert_called_once()


@pytest.mark.asyncio
@patch("ai_platform.llm_gateway.middleware.request_logging.INPUT_TOKENS_TOTAL")
@patch("ai_platform.llm_gateway.middleware.request_logging.OUTPUT_TOKENS_TOTAL")
@patch("ai_platform.llm_gateway.middleware.request_logging.ESTIMATED_COST_TOTAL")
async def test_metrics_are_recorded(
    mock_cost,
    mock_output,
    mock_input,
):

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat",
        "headers": [],
        "client": ("127.0.0.1", 5000),
    }

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"provider":"openai","model":"gpt-4"}',
            "more_body": False,
        }

    request = Request(scope, receive)

    request.state.request_id = "abc123"

    request.state.metrics = {
        "tokens_in": 10,
        "tokens_out": 20,
        "estimated_cost": 0.015,
    }

    response = Response(status_code=200)

    call_next = AsyncMock(return_value=response)

    middleware = RequestLoggingMiddleware(AsyncMock())

    await middleware.dispatch(
        request,
        call_next,
    )

    mock_input.inc.assert_called_once_with(10)
    mock_output.inc.assert_called_once_with(20)
    mock_cost.inc.assert_called_once_with(0.015)
