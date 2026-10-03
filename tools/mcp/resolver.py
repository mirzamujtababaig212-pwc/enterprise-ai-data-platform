from __future__ import annotations

from collections.abc import Sequence

from tools.contracts import ToolRegistry
from tools.models import ToolDefinition


class MCPCapabilityResolutionError(RuntimeError):
    """Raised when an MCP tool capability cannot be resolved unambiguously."""


class MCPCapabilityNotFoundError(MCPCapabilityResolutionError):
    """Raised when no tool or capability matches the requested identifier."""


class MCPCapabilityAmbiguousError(MCPCapabilityResolutionError):
    """Raised when multiple tools match the requested capability."""


class MCPCapabilityResolver:
    """
    Resolves agent-requested MCP tool names or capability names
    against tools currently available in the ToolRegistry.

    Resolution precedence:
    1. Exact tool name.
    2. Unique metadata capability match.

    Disabled tools are already excluded by ToolRegistry.list_tools().
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def resolve_tools_for_agent(
        self,
        requested_names: Sequence[str],
    ) -> list[ToolDefinition]:
        available_tools = await self.registry.list_tools()
        resolved: list[ToolDefinition] = []
        resolved_names: set[str] = set()

        for requested_name in requested_names:
            if not requested_name.strip():
                raise ValueError("Requested tool name must not be empty.")

            exact_match = next(
                (tool for tool in available_tools if tool.name == requested_name),
                None,
            )

            if exact_match is not None:
                if exact_match.name not in resolved_names:
                    resolved.append(exact_match)
                    resolved_names.add(exact_match.name)
                continue

            capability_matches = [
                tool
                for tool in available_tools
                if tool.metadata.get("capability") == requested_name
            ]

            if not capability_matches:
                raise MCPCapabilityNotFoundError(
                    f"No tool or capability matches '{requested_name}'."
                )

            if len(capability_matches) > 1:
                names = ", ".join(tool.name for tool in capability_matches)
                raise MCPCapabilityAmbiguousError(
                    f"Capability '{requested_name}' is ambiguous; " f"matched tools: {names}."
                )

            match = capability_matches[0]

            if match.name not in resolved_names:
                resolved.append(match)
                resolved_names.add(match.name)

        return resolved
