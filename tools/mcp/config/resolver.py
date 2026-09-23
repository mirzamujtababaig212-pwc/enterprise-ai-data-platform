from __future__ import annotations

from tools.mcp.config.models import MCPToolCapability

DEFAULT_MCP_TOOL_CAPABILITY = MCPToolCapability(
    capability="unclassified",
    risk_tier="unknown",
    side_effect=True,
)


def resolve_mcp_tool_capability(
    capability: MCPToolCapability | None,
) -> MCPToolCapability:
    """
    Resolve Deldai governance metadata for an MCP tool.

    Explicit classification always wins. Unclassified MCP tools receive
    a conservative default so that downstream governance policies do not
    accidentally treat unknown external capabilities as low risk.
    """
    if capability is not None:
        return capability

    return DEFAULT_MCP_TOOL_CAPABILITY
