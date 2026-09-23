from __future__ import annotations

from typing import Any

from tools.execution.context import ToolExecutionContext
from tools.execution.idempotency import build_external_idempotency_key
from tools.mcp.client import MCPClient
from tools.mcp.config import MCPToolCapability
from tools.mcp.models import MCPToolDefinition
from tools.models import ToolDefinition


class MCPToolAdapter:
    def __init__(
        self,
        client: MCPClient,
        definition: MCPToolDefinition,
        server_name: str | None = None,
        capability: MCPToolCapability | None = None,
    ):
        if not definition.name.strip():
            raise ValueError("MCP tool name must not be empty.")

        if not definition.description.strip():
            raise ValueError("MCP tool description must not be empty.")

        if server_name is not None and not server_name.strip():
            raise ValueError("MCP server name must not be empty.")

        self.client = client
        self._definition = definition
        self._server_name = server_name
        self._capability = capability

    @property
    def definition(self) -> ToolDefinition:
        metadata = {
            "source": "mcp",
        }

        if self._server_name is not None:
            metadata["mcp_server"] = self._server_name

        if self._capability is not None:
            metadata.update(
                {
                    "capability": self._capability.capability,
                    "risk_tier": self._capability.risk_tier,
                    "side_effect": self._capability.side_effect,
                }
            )

            if self._capability.permission_scope is not None:
                metadata["permission_scope"] = self._capability.permission_scope

        return ToolDefinition(
            name=self._definition.name,
            description=self._definition.description,
            input_schema=dict(self._definition.input_schema),
            metadata=metadata,
        )

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        return await self.execute_with_context(
            arguments,
            None,
        )

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        execution_context: ToolExecutionContext | None,
    ) -> Any:
        if arguments is None:
            raise ValueError("Tool arguments must not be None.")

        meta = None

        if (
            execution_context is not None
            and execution_context.run_id is not None
            and execution_context.call_id is not None
        ):
            meta = {
                "deldai": {
                    "idempotency_key": build_external_idempotency_key(
                        execution_context.run_id,
                        execution_context.call_id,
                        self._definition.name,
                    ),
                }
            }

        if meta is None:
            result = await self.client.call_tool(
                self._definition.name,
                arguments,
            )
        else:
            result = await self.client.call_tool(
                self._definition.name,
                arguments,
                meta=meta,
            )

        if not result.success:
            raise RuntimeError(result.error or "MCP tool execution failed.")

        return result.output
