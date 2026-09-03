from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MCPServerConfig:
    """
    Configuration describing an MCP server.

    The configuration is transport-oriented but does not
    create or manage an MCP client.
    """

    name: str
    transport: str

    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None

    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    timeout: float = 30.0
    read_timeout: float = 300.0
    verify_ssl: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("MCP server name must not be empty.")

        if not self.transport.strip():
            raise ValueError("MCP server transport must not be empty.")

        if self.transport == "stdio":
            if not self.command:
                raise ValueError("MCP stdio server requires a command.")

        elif self.transport == "streamable-http":
            if not self.url:
                raise ValueError("MCP Streamable HTTP server requires a URL.")

        if self.command is not None and not self.command.strip():
            raise ValueError("MCP server command must not be empty.")

        if self.cwd is not None and not self.cwd.strip():
            raise ValueError("MCP server cwd must not be empty.")

        if self.url is not None and not self.url.strip():
            raise ValueError("MCP server URL must not be empty.")

        if any(not argument for argument in self.args):
            raise ValueError("MCP server arguments must not contain empty values.")

        if any(not key.strip() for key in self.env):
            raise ValueError("MCP server environment variable names " "must not be empty.")

        if any(not key.strip() for key in self.headers):
            raise ValueError("MCP server HTTP header names must not be empty.")

        if self.timeout <= 0:
            raise ValueError("MCP HTTP timeout must be greater than zero.")

        if self.read_timeout <= 0:
            raise ValueError("MCP HTTP read timeout must be greater than zero.")
