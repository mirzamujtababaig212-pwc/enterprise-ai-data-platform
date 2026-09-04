from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Sequence

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tools.mcp.client import MCPClient
from tools.mcp.models import (
    MCPToolCallResult,
    MCPToolDefinition,
)


class MCPPythonSDKClient(MCPClient):
    """
    MCP client implementation backed by the official
    MCP Python SDK v1.x.

    The rest of the Enterprise AI platform depends only on
    the internal MCPClient protocol.

    Currently supports:
        - stdio MCP servers
    """

    def __init__(
        self,
        server: StdioServerParameters,
    ) -> None:
        self.server = server

        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._connected = False

    async def connect(self) -> None:
        """
        Open the stdio transport and initialize the MCP session.
        """

        if self._connected:
            return

        exit_stack = AsyncExitStack()

        try:
            read, write = await exit_stack.enter_async_context(stdio_client(self.server))

            session = await exit_stack.enter_async_context(ClientSession(read, write))

            await session.initialize()

            self._exit_stack = exit_stack
            self._session = session
            self._connected = True

        except Exception:
            await exit_stack.aclose()
            raise

    async def disconnect(self) -> None:
        """
        Close the MCP session and underlying transport.
        """

        if self._exit_stack is None:
            self._session = None
            self._connected = False
            return

        try:
            await self._exit_stack.aclose()
        finally:
            self._exit_stack = None
            self._session = None
            self._connected = False

    async def list_tools(
        self,
    ) -> Sequence[MCPToolDefinition]:
        """
        Discover tools exposed by the MCP server.
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
        Invoke a tool exposed by the MCP server.
        """

        if not name.strip():
            raise ValueError("MCP tool name must not be empty.")

        if arguments is None:
            raise ValueError("Tool arguments must not be None.")

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
                "MCP client is not connected. " "Call connect() before using the client."
            )

        return self._session

    @staticmethod
    def _extract_output(result: Any) -> Any:
        structured = getattr(result, "structuredContent", None)

        if structured is not None:
            return structured

        content = getattr(result, "content", None)

        if not content:
            return None

        extracted: list[Any] = []

        for block in content:
            text = getattr(block, "text", None)

            if text is not None:
                try:
                    import json

                    extracted.append(json.loads(text))
                except (json.JSONDecodeError, TypeError):
                    extracted.append(text)
            else:
                extracted.append(block)

        if len(extracted) == 1:
            return extracted[0]

        return extracted

    @staticmethod
    def _extract_error(result: Any) -> str:
        """
        Extract a useful error message from an MCP error result.
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
