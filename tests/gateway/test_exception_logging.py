from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Request
from starlette.responses import Response

from ai_platform.llm_gateway.middleware.exception_logging import (
    ExceptionLoggingMiddleware,
)


@pytest.mark.asyncio
async def test_dispatch_success():

    app = MagicMock()

    middleware = ExceptionLoggingMiddleware(app)

    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.request_id = "req-123"

    response = Response(status_code=200)

    call_next = AsyncMock(return_value=response)

    result = await middleware.dispatch(
        request,
        call_next,
    )

    assert result.status_code == 200


@pytest.mark.asyncio
@patch("ai_platform.llm_gateway.middleware.exception_logging.logger")
async def test_dispatch_exception(
    mock_logger,
):

    app = MagicMock()

    middleware = ExceptionLoggingMiddleware(app)

    request = MagicMock(spec=Request)

    request.state = MagicMock()
    request.state.request_id = "req-456"

    request.method = "POST"

    request.url.path = "/v1/chat"

    request.client.host = "127.0.0.1"

    async def failing_call_next(request):
        raise RuntimeError("boom")

    response = await middleware.dispatch(
        request,
        failing_call_next,
    )

    assert response.status_code == 500

    body = response.body.decode()

    assert "Internal Server Error" in body

    mock_logger.exception.assert_called_once()
