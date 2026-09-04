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

FIXTURES_DIR = PROJECT_ROOT / "tests" / "tools" / "mcp" / "fixtures"

SERVERS_DIR = PROJECT_ROOT / "tests" / "tools" / "mcp" / "servers"

STDIO_SERVER = FIXTURES_DIR / "test_server.py"

HTTP_SERVER = SERVERS_DIR / "streamable_http_server.py"


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
    Wait until a TCP server accepts connections.
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

    raise TimeoutError(f"Timed out waiting for server at {host}:{port}.")


@pytest.fixture
def streamable_http_server() -> Iterator[str]:
    """
    Start the real FastMCP Streamable HTTP test server.
    """
    host = "127.0.0.1"
    port = find_free_port()

    process = subprocess.Popen(
        [
            sys.executable,
            str(HTTP_SERVER),
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
async def test_manager_manages_real_stdio_and_streamable_http_servers(
    streamable_http_server: str,
):
    """
    Verify that MCPServerManager can simultaneously manage:

        Real STDIO MCP server
                +
        Real Streamable HTTP MCP server

    using a single ToolRegistry.

    Complete path:

        MCPServerManager
             |
             +----------------------+
             |                      |
             v                      v
        STDIO server           HTTP server
             |                      |
             v                      v
       STDIO client           HTTP client
             |                      |
             +----------+-----------+
                        |
                        v
                 Tool discovery
                        |
                        v
                  Tool adapters
                        |
                        v
                Tool registry
                        |
             +----------+----------+
             |                     |
             v                     v
       search_documents          echo
             |                  add_numbers
             |              get_server_identity
             v                     |
        Real execution       Real execution
    """

    registry = InMemoryToolRegistry()

    manager = MCPServerManager(
        registry=registry,
    )

    stdio_config = MCPServerConfig(
        name="document-server",
        transport="stdio",
        command=sys.executable,
        args=(str(STDIO_SERVER),),
    )

    http_config = MCPServerConfig(
        name="http-server",
        transport="streamable-http",
        url=streamable_http_server,
    )

    # ---------------------------------------------------------
    # 1. Register both MCP servers.
    # ---------------------------------------------------------
    await manager.register_server(
        stdio_config,
    )

    await manager.register_server(
        http_config,
    )

    assert manager.list_servers() == [
        "document-server",
        "http-server",
    ]

    assert manager.is_connected("document-server") is False

    assert manager.is_connected("http-server") is False

    # ---------------------------------------------------------
    # 2. Connect and discover the STDIO server.
    # ---------------------------------------------------------
    try:
        stdio_definitions = await manager.connect_and_discover("document-server")

        assert manager.is_connected("document-server") is True

        assert len(stdio_definitions) == 1

        assert stdio_definitions[0].name == ("search_documents")

        assert stdio_definitions[0].description.strip() == (
            "Search a deterministic test document collection."
        )

        # -----------------------------------------------------
        # 3. Connect and discover the HTTP server.
        # -----------------------------------------------------
        http_definitions = await manager.connect_and_discover("http-server")

        assert manager.is_connected("http-server") is True

        http_tool_names = {definition.name for definition in http_definitions}

        assert http_tool_names == {
            "echo",
            "add_numbers",
            "get_server_identity",
        }

        # -----------------------------------------------------
        # 4. Verify both transports coexist in the same
        #    ToolRegistry.
        # -----------------------------------------------------
        registered_tools = await registry.list_tools()

        registered_names = {definition.name for definition in registered_tools}

        assert registered_names == {
            "search_documents",
            "echo",
            "add_numbers",
            "get_server_identity",
        }

        # -----------------------------------------------------
        # 5. Execute a real STDIO MCP tool through the
        #    platform ToolRegistry.
        # -----------------------------------------------------
        search_tool = await registry.get(
            "search_documents",
        )

        assert search_tool is not None

        search_result = await search_tool.execute(
            {
                "query": "enterprise AI",
            }
        )

        assert search_result["query"] == ("enterprise AI")

        assert search_result["results"] == [
            {
                "id": "document-1",
                "content": ("Enterprise AI platform architecture."),
            }
        ]

        # -----------------------------------------------------
        # 6. Execute a real Streamable HTTP MCP tool.
        # -----------------------------------------------------
        echo_tool = await registry.get(
            "echo",
        )

        assert echo_tool is not None

        echo_result = await echo_tool.execute(
            {
                "message": "multi-transport test",
            }
        )

        assert echo_result == {
            "result": "multi-transport test",
        }

        # -----------------------------------------------------
        # 7. Execute another real HTTP MCP tool.
        # -----------------------------------------------------
        add_tool = await registry.get(
            "add_numbers",
        )

        assert add_tool is not None

        add_result = await add_tool.execute(
            {
                "a": 40,
                "b": 2,
            }
        )

        assert add_result == {
            "result": 42,
        }

        # -----------------------------------------------------
        # 8. Execute the HTTP server identity tool.
        # -----------------------------------------------------
        identity_tool = await registry.get(
            "get_server_identity",
        )

        assert identity_tool is not None

        identity_result = await identity_tool.execute({})

        assert identity_result == {
            "server": "enterprise-ai-platform-test-server",
            "transport": "streamable-http",
        }

        # -----------------------------------------------------
        # 9. Verify both servers remain connected while
        #    tools from both transports are being used.
        # -----------------------------------------------------
        assert manager.is_connected("document-server") is True

        assert manager.is_connected("http-server") is True

    finally:
        # ---------------------------------------------------------
        # 10. Disconnect all MCP servers through the manager.
        # ---------------------------------------------------------
        await manager.disconnect_all()

    # ---------------------------------------------------------
    # 11. Verify both transports were cleanly disconnected.
    # ---------------------------------------------------------
    assert manager.is_connected("document-server") is False

    assert manager.is_connected("http-server") is False
