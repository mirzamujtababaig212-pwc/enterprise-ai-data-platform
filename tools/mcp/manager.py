from __future__ import annotations

from dataclasses import dataclass

from mcp import StdioServerParameters

from tools.contracts import ToolRegistry
from tools.mcp.client import MCPClient
from tools.mcp.config import MCPServerConfig
from tools.mcp.discovery import MCPToolDiscoveryService
from tools.mcp.http_client import MCPStreamableHTTPClient
from tools.mcp.sdk_client import MCPPythonSDKClient
from tools.models import ToolDefinition


@dataclass
class _MCPServerRuntime:
    """
    Internal runtime state for a registered MCP server.
    """

    config: MCPServerConfig
    client: MCPClient
    connected: bool = False


class MCPServerManager:
    """
    Coordinates the lifecycle of multiple MCP servers.

    Responsibilities:
    - register MCP server configurations
    - create MCP clients from configurations
    - connect and disconnect servers
    - discover and register tools
    - track server runtime state
    - clean up all active MCP connections

    Transport-specific protocol behavior remains inside the
    corresponding MCPClient implementation.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._servers: dict[str, _MCPServerRuntime] = {}

    async def register_server(
        self,
        config: MCPServerConfig,
    ) -> None:
        """
        Register a new MCP server configuration.

        Registration does not establish a connection.
        """
        if config.name in self._servers:
            raise ValueError(f"MCP server '{config.name}' is already registered.")

        client = self._create_client(config)

        self._servers[config.name] = _MCPServerRuntime(
            config=config,
            client=client,
        )

    async def connect_server(
        self,
        name: str,
    ) -> None:
        """
        Connect a registered MCP server.
        """
        runtime = self._get_runtime(name)

        if runtime.connected:
            return

        try:
            connect = getattr(
                runtime.client,
                "connect",
                None,
            )

            if connect is None:
                raise RuntimeError(f"MCP client for server '{name}' " "does not support connect().")

            await connect()
            runtime.connected = True

        except Exception:
            runtime.connected = False
            raise

    async def discover_server(
        self,
        name: str,
    ) -> list[ToolDefinition]:
        """
        Discover tools from a connected MCP server and register them
        with the platform ToolRegistry.
        """
        runtime = self._get_runtime(name)

        if not runtime.connected:
            raise RuntimeError(
                f"MCP server '{name}' is not connected. "
                "Call connect_server() before discover_server()."
            )

        discovery = MCPToolDiscoveryService(
            client=runtime.client,
            registry=self.registry,
        )

        return await discovery.discover_and_register()

    async def connect_and_discover(
        self,
        name: str,
    ) -> list[ToolDefinition]:
        """
        Connect to an MCP server and discover/register its tools.
        """
        await self.connect_server(name)

        return await self.discover_server(name)

    async def disconnect_server(
        self,
        name: str,
    ) -> None:
        """
        Disconnect a registered MCP server.

        Disconnecting an already-disconnected server is a no-op.
        """
        runtime = self._get_runtime(name)

        if not runtime.connected:
            return

        try:
            disconnect = getattr(
                runtime.client,
                "disconnect",
                None,
            )

            if disconnect is None:
                raise RuntimeError(
                    f"MCP client for server '{name}' " "does not support disconnect()."
                )

            await disconnect()

        finally:
            runtime.connected = False

    async def disconnect_all(self) -> None:
        """
        Disconnect every registered MCP server in reverse
        registration order.

        MCP stdio clients own subprocess and AnyIO resources whose
        cleanup is safest when performed in LIFO order.

        All servers are attempted even if one disconnect operation
        fails. The first encountered exception is re-raised after
        cleanup attempts.
        """
        first_error: Exception | None = None

        for name in reversed(list(self._servers)):
            try:
                await self.disconnect_server(name)

            except Exception as exc:
                if first_error is None:
                    first_error = exc

        if first_error is not None:
            raise first_error

    async def get_client(
        self,
        name: str,
    ) -> MCPClient:
        """
        Return the client associated with a registered server.
        """
        return self._get_runtime(name).client

    def get_config(
        self,
        name: str,
    ) -> MCPServerConfig:
        """
        Return the configuration associated with a registered server.
        """
        return self._get_runtime(name).config

    def is_connected(
        self,
        name: str,
    ) -> bool:
        """
        Return whether a registered MCP server is currently connected.
        """
        return self._get_runtime(name).connected

    def list_servers(self) -> list[str]:
        """
        Return the registered MCP server names.
        """
        return list(self._servers.keys())

    def _get_runtime(
        self,
        name: str,
    ) -> _MCPServerRuntime:
        if not name.strip():
            raise ValueError("MCP server name must not be empty.")

        runtime = self._servers.get(name)

        if runtime is None:
            raise KeyError(f"MCP server '{name}' is not registered.")

        return runtime

    @staticmethod
    def _create_client(
        config: MCPServerConfig,
    ) -> MCPClient:
        """
        Create the transport-specific MCP client for a configuration.
        """
        if config.transport == "stdio":
            if config.command is None:
                raise ValueError("MCP stdio server requires a command.")

            server_parameters = StdioServerParameters(
                command=config.command,
                args=list(config.args),
                env=(dict(config.env) if config.env else None),
                cwd=config.cwd,
            )

            return MCPPythonSDKClient(server_parameters)

        if config.transport == "streamable-http":
            if config.url is None:
                raise ValueError("MCP Streamable HTTP server requires a URL.")

            return MCPStreamableHTTPClient(
                config.url,
                headers=config.headers,
                timeout=config.timeout,
                read_timeout=config.read_timeout,
                verify=config.verify_ssl,
            )

        raise ValueError(f"Unsupported MCP server transport: " f"'{config.transport}'.")
