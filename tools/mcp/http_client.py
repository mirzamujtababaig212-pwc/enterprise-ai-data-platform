from __future__ import annotations

import json
from contextlib import AsyncExitStack
from typing import Any, Sequence

import httpx

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from tools.mcp.client import MCPClient
from tools.mcp.models import MCPToolCallResult, MCPToolDefinition


class MCPStreamableHTTPClient(MCPClient):
    """
    MCP client implementation using Streamable HTTP transport.

    The client owns:
    - the HTTP client
    - the Streamable HTTP transport
    - the MCP ClientSession

    Transport-specific details remain hidden behind the MCPClient
    protocol so discovery and tool execution remain transport-agnostic.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        read_timeout: float = 300.0,
        verify: bool = True,
    ) -> None:
        if not url.strip():
            raise ValueError("MCP Streamable HTTP URL must not be empty.")

        if timeout <= 0:
            raise ValueError("MCP HTTP timeout must be greater than zero.")

        if read_timeout <= 0:
            raise ValueError("MCP HTTP read timeout must be greater than zero.")

        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.read_timeout = read_timeout
        self.verify = verify

        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._connected = False

    async def connect(self) -> None:
        """
        Establish the Streamable HTTP transport and initialize the MCP
        session.
        """
        if self._connected:
            return

        exit_stack = AsyncExitStack()

        try:
            timeout = httpx.Timeout(
                self.timeout,
                read=self.read_timeout,
            )

            http_client = httpx.AsyncClient(
                headers=self.headers,
                timeout=timeout,
                follow_redirects=True,
                verify=self.verify,
            )

            await exit_stack.enter_async_context(http_client)

            transport = streamable_http_client(
                self.url,
                http_client=http_client,
            )

            read_stream, write_stream, _ = await exit_stack.enter_async_context(transport)

            session = await exit_stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                )
            )

            await session.initialize()

            self._http_client = http_client
            self._session = session
            self._exit_stack = exit_stack
            self._connected = True

        except Exception:
            await exit_stack.aclose()
            raise

    async def disconnect(self) -> None:
        """
        Close the MCP session and underlying HTTP transport.
        """
        if self._exit_stack is None:
            self._session = None
            self._http_client = None
            self._connected = False
            return

        try:
            await self._exit_stack.aclose()

        finally:
            self._exit_stack = None
            self._session = None
            self._http_client = None
            self._connected = False

    async def list_tools(self) -> Sequence[MCPToolDefinition]:
        """
        Retrieve tools exposed by the remote MCP server.
        """
        session = self._require_session()

        result = await session.list_tools()

        definitions: list[MCPToolDefinition] = []

        for tool in result.tools:
            definitions.append(
                MCPToolDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(tool.inputSchema if tool.inputSchema is not None else {}),
                )
            )

        return definitions

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> MCPToolCallResult:
        """
        Execute an MCP tool over Streamable HTTP.
        """
        if not name.strip():
            raise ValueError("MCP tool name must not be empty.")

        if arguments is None:
            raise ValueError("MCP tool arguments must not be None.")

        session = self._require_session()

        try:
            result = await session.call_tool(
                name,
                arguments=arguments,
            )

            if result.isError:
                return MCPToolCallResult(error=self._extract_error(result))

            return MCPToolCallResult(output=self._extract_output(result))

        except Exception as exc:
            return MCPToolCallResult(error=f"{type(exc).__name__}: {exc}")

    def _require_session(self) -> ClientSession:
        if self._session is None or not self._connected:
            raise RuntimeError(
                "MCP HTTP client is not connected. " "Call connect() before using the client."
            )

        return self._session

    @staticmethod
    def _extract_output(result: Any) -> Any:
        """
        Normalize MCP tool output into ordinary Python values.
        """
        structured = getattr(
            result,
            "structuredContent",
            None,
        )

        if structured is not None:
            return structured

        content = getattr(
            result,
            "content",
            None,
        )

        if not content:
            return None

        extracted: list[Any] = []

        for block in content:
            text = getattr(
                block,
                "text",
                None,
            )

            if text is not None:
                try:
                    extracted.append(json.loads(text))
                except (
                    json.JSONDecodeError,
                    TypeError,
                ):
                    extracted.append(text)

            else:
                extracted.append(block)

        if len(extracted) == 1:
            return extracted[0]

        return extracted

    @staticmethod
    def _extract_error(result: Any) -> str:
        """
        Extract human-readable MCP tool errors.
        """
        content = getattr(
            result,
            "content",
            None,
        )

        if content:
            messages: list[str] = []

            for block in content:
                text = getattr(
                    block,
                    "text",
                    None,
                )

                if text:
                    messages.append(text)

            if messages:
                return " | ".join(messages)

        return "MCP tool execution failed."
