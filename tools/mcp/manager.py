from __future__ import annotations

import asyncio
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
import time

from mcp import StdioServerParameters

from tools.contracts import ToolRegistry
from tools.mcp.client import MCPClient
from tools.mcp.config import MCPServerConfig
from tools.mcp.discovery import MCPToolDiscoveryService
from tools.mcp.health import MCPHealthHistory, MCPHealthStatus
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
    health_history: MCPHealthHistory = dataclass_field(
        default_factory=MCPHealthHistory,
    )
    recovery_lock: asyncio.Lock = dataclass_field(
        default_factory=asyncio.Lock,
    )
    last_recovery_at: datetime | None = None
    discovered_tool_names: set[str] = dataclass_field(default_factory=set)


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
            server_name=name,
            tool_capabilities=runtime.config.tool_capabilities,
        )

        definitions = await discovery.discover_and_register()
        runtime.discovered_tool_names = {definition.name for definition in definitions}
        return definitions

    async def connect_and_discover(
        self,
        name: str,
    ) -> list[ToolDefinition]:
        """
        Connect to an MCP server and discover/register its tools.
        """
        await self.connect_server(name)

        return await self.discover_server(name)

    async def recover_server(
        self,
        name: str,
    ) -> list[ToolDefinition]:
        """
        Explicitly recover an MCP server using its bounded recovery policy.

        Recovery is serialized per server and never runs in the background.
        Each attempt disconnects any stale connection, reconnects the server,
        verifies protocol liveness, and re-discovers its current tools.
        """
        runtime = self._get_runtime(name)
        policy = runtime.config.recovery_policy

        async with runtime.recovery_lock:
            now = datetime.now(timezone.utc)

            if (
                runtime.last_recovery_at is not None
                and policy.cooldown > 0
                and (now - runtime.last_recovery_at).total_seconds() < policy.cooldown
            ):
                remaining = policy.cooldown - (now - runtime.last_recovery_at).total_seconds()
                raise RuntimeError(
                    f"MCP server '{name}' recovery is in cooldown "
                    f"for {remaining:.2f} more seconds."
                )

            runtime.last_recovery_at = now
            last_error: Exception | None = None

            for attempt in range(1, policy.max_attempts + 1):
                if not policy.allows_attempt(attempt):
                    break

                if attempt > 1:
                    await asyncio.sleep(policy.backoff_for_attempt(attempt - 1))

                try:
                    await self.disconnect_server(name)
                    await self.connect_server(name)

                    await self._verify_recovery_health(name)

                    return await self.discover_server(name)

                except Exception as exc:
                    last_error = exc
                    runtime.connected = False

            if last_error is None:
                raise RuntimeError(f"MCP server '{name}' recovery failed without an error.")

            raise RuntimeError(
                f"MCP server '{name}' recovery failed after " f"{policy.max_attempts} attempt(s)."
            ) from last_error

    async def _verify_recovery_health(
        self,
        name: str,
    ) -> None:
        runtime = self._get_runtime(name)

        send_ping = getattr(
            runtime.client,
            "send_ping",
            None,
        )

        if send_ping is None:
            raise RuntimeError(f"MCP client for server '{name}' does not support protocol ping.")

        await asyncio.wait_for(
            send_ping(),
            timeout=runtime.config.health_check_timeout,
        )

        checked_at = datetime.now(timezone.utc)
        runtime.health_history.record_success(checked_at)

    async def check_health(
        self,
        name: str,
    ) -> MCPHealthStatus:
        """
        Check the liveness of a registered MCP server.

        Health checking is observational only:
        - it never reconnects the server
        - it never performs tool discovery
        - it never executes an MCP tool

        The MCP protocol ping is used when supported by the client.
        """
        runtime = self._get_runtime(name)
        checked_at = datetime.now(timezone.utc)

        if not runtime.connected:
            status = runtime.health_history.record_failure(
                checked_at,
                error="MCP server is not connected.",
                hard_failure=True,
            )

            return MCPHealthStatus(
                server_name=name,
                status=status,
                latency_ms=0.0,
                last_check=checked_at,
                error="MCP server is not connected.",
            )

        send_ping = getattr(
            runtime.client,
            "send_ping",
            None,
        )

        if send_ping is None:
            status = runtime.health_history.record_failure(
                checked_at,
                error="MCP client does not support protocol ping.",
                hard_failure=True,
            )

            return MCPHealthStatus(
                server_name=name,
                status=status,
                latency_ms=0.0,
                last_check=checked_at,
                error="MCP client does not support protocol ping.",
            )

        started = time.perf_counter()

        try:
            await asyncio.wait_for(
                send_ping(),
                timeout=runtime.config.health_check_timeout,
            )

        except asyncio.TimeoutError:
            checked_at = datetime.now(timezone.utc)
            error = (
                "MCP health check timed out after "
                f"{runtime.config.health_check_timeout:g} seconds."
            )
            status = runtime.health_history.record_failure(
                checked_at,
                error=error,
            )

            return MCPHealthStatus(
                server_name=name,
                status=status,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                last_check=checked_at,
                error=error,
            )

        except Exception as exc:
            checked_at = datetime.now(timezone.utc)
            error = f"{type(exc).__name__}: {exc}"
            status = runtime.health_history.record_failure(
                checked_at,
                error=error,
            )

            return MCPHealthStatus(
                server_name=name,
                status=status,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                last_check=checked_at,
                error=error,
            )

        checked_at = datetime.now(timezone.utc)
        status = runtime.health_history.record_success(checked_at)

        return MCPHealthStatus(
            server_name=name,
            status=status,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            last_check=checked_at,
        )

    async def check_all_health(self) -> dict[str, MCPHealthStatus]:
        """
        Check the liveness of every registered MCP server.

        Each server is checked independently. A failure for one server
        does not prevent health observations for the remaining servers.
        """
        statuses: dict[str, MCPHealthStatus] = {}

        for name in self._servers:
            statuses[name] = await self.check_health(name)

        return statuses

    async def unregister_server(
        self,
        name: str,
    ) -> None:
        """
        Unregister an MCP server and remove its discovered tools.

        If the server is connected, it is disconnected before its runtime
        registration is removed. A disconnect failure leaves the server
        registered so callers do not observe a false successful removal.

        Only tools explicitly owned by this MCP server are removed.
        """
        self._get_runtime(name)

        await self.disconnect_server(name)

        runtime = self._get_runtime(name)

        for tool_name in runtime.discovered_tool_names:
            await self.registry.remove(tool_name)

        self._servers.pop(name, None)

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
