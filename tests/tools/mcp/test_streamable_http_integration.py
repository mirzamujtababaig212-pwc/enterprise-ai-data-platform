from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest

from tools.mcp.config import MCPServerConfig
from tools.mcp.manager import MCPServerManager
from tools.registry.in_memory import InMemoryToolRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SERVER_SCRIPT = PROJECT_ROOT / "tests" / "tools" / "mcp" / "servers" / "streamable_http_server.py"


def find_free_port() -> int:
    """
    Ask the operating system for an available localhost TCP port.
    """
    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM,
    ) as sock:
        sock.bind(("127.0.0.1", 0))

        return int(sock.getsockname()[1])


def wait_for_port(
    host: str,
    port: int,
    *,
    timeout: float = 10.0,
) -> None:
    """
    Wait until the MCP HTTP server accepts TCP connections.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            with socket.create_connection(
                (host, port),
                timeout=0.5,
            ):
                return

        except OSError:
            time.sleep(0.1)

    raise TimeoutError(f"Timed out waiting for MCP server at {host}:{port}.")


@pytest.fixture
def streamable_http_server() -> Iterator[str]:
    """
    Start a real FastMCP Streamable HTTP server and
    terminate it after the test.
    """
    host = "127.0.0.1"
    port = find_free_port()

    process = subprocess.Popen(
        [
            sys.executable,
            str(SERVER_SCRIPT),
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        wait_for_port(
            host,
            port,
        )

        yield f"http://{host}:{port}/mcp"

    except Exception:
        if process.poll() is None:
            process.terminate()

        try:
            stdout, stderr = process.communicate(
                timeout=5,
            )
        except subprocess.TimeoutExpired:
            process.kill()

            stdout, stderr = process.communicate(
                timeout=5,
            )

        pytest.fail(
            "Real Streamable HTTP MCP test server failed.\n\n"
            f"stdout:\n{stdout}\n\n"
            f"stderr:\n{stderr}"
        )

    finally:
        if process.poll() is None:
            process.terminate()

            try:
                process.wait(
                    timeout=5,
                )
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(
                    timeout=5,
                )


@pytest.mark.asyncio
async def test_real_streamable_http_server_tool_discovery_and_execution(
    streamable_http_server: str,
):
    """
    Verify the complete real Streamable HTTP MCP path:

        FastMCP server
            ↓
        Streamable HTTP
            ↓
        MCPStreamableHTTPClient
            ↓
        MCPServerManager
            ↓
        MCPToolDiscoveryService
            ↓
        MCPToolAdapter
            ↓
        InMemoryToolRegistry
            ↓
        Tool execution
    """
    registry = InMemoryToolRegistry()

    manager = MCPServerManager(
        registry=registry,
    )

    config = MCPServerConfig(
        name="real-http-server",
        transport="streamable-http",
        url=streamable_http_server,
    )

    try:
        # ---------------------------------------------------------
        # 1. Register the MCP server.
        # ---------------------------------------------------------
        await manager.register_server(
            config,
        )

        assert manager.list_servers() == [
            "real-http-server",
        ]

        assert manager.is_connected("real-http-server") is False

        # ---------------------------------------------------------
        # 2. Connect and discover its tools.
        # ---------------------------------------------------------
        definitions = await manager.connect_and_discover(
            "real-http-server",
        )

        assert manager.is_connected("real-http-server") is True

        # ---------------------------------------------------------
        # 3. Verify discovered tool definitions.
        # ---------------------------------------------------------
        tool_names = {definition.name for definition in definitions}

        assert "echo" in tool_names
        assert "add_numbers" in tool_names
        assert "get_server_identity" in tool_names

        assert len(definitions) == 3

        # ---------------------------------------------------------
        # 4. Verify that the tools were registered in the
        #    actual platform ToolRegistry.
        # ---------------------------------------------------------
        registered_tools = await registry.list_tools()

        registered_tool_names = {definition.name for definition in registered_tools}

        assert "echo" in registered_tool_names
        assert "add_numbers" in registered_tool_names
        assert "get_server_identity" in registered_tool_names

        # ---------------------------------------------------------
        # 5. Execute the real MCP "echo" tool through the
        #    platform ToolRegistry.
        # ---------------------------------------------------------
        echo_tool = await registry.get(
            "echo",
        )

        assert echo_tool is not None

        echo_result = await echo_tool.execute(
            {
                "message": "hello enterprise AI",
            }
        )

        assert echo_result == {
            "result": "hello enterprise AI",
        }

        # ---------------------------------------------------------
        # 6. Execute the real MCP "add_numbers" tool.
        # ---------------------------------------------------------
        add_tool = await registry.get(
            "add_numbers",
        )

        assert add_tool is not None

        add_result = await add_tool.execute(
            {
                "a": 10,
                "b": 32,
            }
        )

        assert add_result == {
            "result": 42,
        }

        # ---------------------------------------------------------
        # 7. Execute the real MCP identity tool.
        # ---------------------------------------------------------
        identity_tool = await registry.get(
            "get_server_identity",
        )

        assert identity_tool is not None

        identity_result = await identity_tool.execute({})

        assert identity_result == {
            "server": "enterprise-ai-platform-test-server",
            "transport": "streamable-http",
        }

    finally:
        # ---------------------------------------------------------
        # 8. Verify clean MCP lifecycle shutdown.
        # ---------------------------------------------------------
        await manager.disconnect_all()

        assert manager.is_connected("real-http-server") is False
