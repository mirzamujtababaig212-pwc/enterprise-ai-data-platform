from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise AI Platform Streamable HTTP MCP test server."
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the MCP test server.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the MCP test server.",
    )

    return parser.parse_args()


def create_server(
    host: str,
    port: int,
) -> FastMCP:
    mcp = FastMCP(
        "enterprise-ai-platform-test-server",
        host=host,
        port=port,
    )

    @mcp.tool()
    def echo(message: str) -> str:
        """Echo a message back to the caller."""
        return message

    @mcp.tool()
    def add_numbers(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @mcp.tool()
    def get_server_identity() -> dict[str, str]:
        """Return identifying information about the test MCP server."""
        return {
            "server": "enterprise-ai-platform-test-server",
            "transport": "streamable-http",
        }

    return mcp


def main() -> None:
    args = parse_args()

    mcp = create_server(
        host=args.host,
        port=args.port,
    )

    mcp.run(
        transport="streamable-http",
    )


if __name__ == "__main__":
    main()
