from tools.mcp.config import (
    DEFAULT_MCP_TOOL_CAPABILITY,
    MCPToolCapability,
    resolve_mcp_tool_capability,
)


def test_resolver_returns_explicit_capability():
    capability = MCPToolCapability(
        capability="document.search",
        risk_tier="low",
        side_effect=False,
        permission_scope="documents:read",
    )

    resolved = resolve_mcp_tool_capability(capability)

    assert resolved is capability


def test_resolver_returns_conservative_default_for_unclassified_tool():
    resolved = resolve_mcp_tool_capability(None)

    assert resolved == DEFAULT_MCP_TOOL_CAPABILITY
    assert resolved.capability == "unclassified"
    assert resolved.risk_tier == "unknown"
    assert resolved.side_effect is True
    assert resolved.permission_scope is None


def test_default_capability_is_not_low_risk():
    assert DEFAULT_MCP_TOOL_CAPABILITY.risk_tier == "unknown"
    assert DEFAULT_MCP_TOOL_CAPABILITY.side_effect is True
