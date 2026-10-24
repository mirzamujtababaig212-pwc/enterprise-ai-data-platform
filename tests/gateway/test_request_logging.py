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


@patch(
    "ai_platform.llm_gateway.middleware.request_logging.logger",
)
@pytest.mark.asyncio
async def test_chat_content_is_not_written_to_request_logs(
    mock_logger,
    middleware,
):
    secret_prompt = "private voice transcript: customer account 12345"
    secret_message = "another private conversation detail"

    payload = {
        "provider": "openai",
        "model": "gpt-4o",
        "prompt": secret_prompt,
        "messages": [
            {"role": "user", "content": secret_message},
        ],
    }
    request = make_request(body=json.dumps(payload).encode())
    response = Response(status_code=200)

    await middleware.dispatch(
        request,
        AsyncMock(return_value=response),
    )

    completed_logs = [
        call
        for call in mock_logger.info.call_args_list
        if call.args and call.args[0] == "HTTP Request Completed"
    ]
    assert len(completed_logs) == 1

    logged_body = completed_logs[0].kwargs["extra"]["request_body"]
    assert logged_body["provider"] == "openai"
    assert logged_body["model"] == "gpt-4o"
    assert logged_body["prompt"] == "********"
    assert logged_body["messages"] == "********"

    serialized_log_body = json.dumps(logged_body)
    assert secret_prompt not in serialized_log_body
    assert secret_message not in serialized_log_body
