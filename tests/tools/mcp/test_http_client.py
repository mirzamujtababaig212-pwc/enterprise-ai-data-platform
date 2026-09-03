from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tools.mcp.http_client import MCPStreamableHTTPClient


def test_http_client_rejects_empty_url():
    with pytest.raises(
        ValueError,
        match="URL must not be empty",
    ):
        MCPStreamableHTTPClient("")


def test_http_client_rejects_invalid_timeout():
    with pytest.raises(
        ValueError,
        match="timeout must be greater than zero",
    ):
        MCPStreamableHTTPClient(
            "http://localhost:8000/mcp",
            timeout=0,
        )


def test_http_client_rejects_invalid_read_timeout():
    with pytest.raises(
        ValueError,
        match="read timeout must be greater than zero",
    ):
        MCPStreamableHTTPClient(
            "http://localhost:8000/mcp",
            read_timeout=0,
        )


@pytest.mark.asyncio
async def test_http_client_connects_and_disconnects():
    client = MCPStreamableHTTPClient("http://localhost:8000/mcp")

    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()

    mock_http_client = MagicMock()

    mock_transport = MagicMock()

    mock_read_stream = MagicMock()
    mock_write_stream = MagicMock()
    mock_get_session_id = MagicMock(return_value=None)

    mock_transport_result = (
        mock_read_stream,
        mock_write_stream,
        mock_get_session_id,
    )

    mock_exit_stack = MagicMock()

    mock_exit_stack.enter_async_context = AsyncMock(
        side_effect=[
            mock_http_client,
            mock_transport_result,
            mock_session,
        ]
    )

    mock_exit_stack.aclose = AsyncMock()

    with (
        patch(
            "tools.mcp.http_client.AsyncExitStack",
            return_value=mock_exit_stack,
        ),
        patch(
            "tools.mcp.http_client.streamable_http_client",
            return_value=mock_transport,
        ),
        patch(
            "tools.mcp.http_client.httpx.AsyncClient",
            return_value=mock_http_client,
        ),
    ):
        await client.connect()

    assert client._connected is True
    assert client._session is mock_session
    assert client._http_client is mock_http_client

    mock_session.initialize.assert_awaited_once()

    await client.disconnect()

    assert client._connected is False
    assert client._session is None
    assert client._http_client is None

    mock_exit_stack.aclose.assert_awaited_once()
