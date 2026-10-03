import pytest

from tools.mcp.resolver import (
    MCPCapabilityAmbiguousError,
    MCPCapabilityNotFoundError,
    MCPCapabilityResolver,
)
from tools.models import ToolDefinition

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


class StubToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self.definitions = definitions

    async def list_tools(self) -> list[ToolDefinition]:
        return [definition for definition in self.definitions if definition.enabled]


def _tool(
    name: str,
    *,
    capability: str | None = None,
    enabled: bool = True,
) -> ToolDefinition:
    metadata = {"source": "mcp"}

    if capability is not None:
        metadata["capability"] = capability

    return ToolDefinition(
        name=name,
        description=f"Test tool {name}",
        metadata=metadata,
        enabled=enabled,
    )


@pytest.mark.asyncio
async def test_capability_resolver_resolves_exact_tool_name():
    search = _tool(
        "documents.search",
        capability="document.search",
    )
    registry = StubToolRegistry([search])
    resolver = MCPCapabilityResolver(registry)

    resolved = await resolver.resolve_tools_for_agent(
        ["documents.search"],
    )

    assert resolved == [search]


@pytest.mark.asyncio
async def test_capability_resolver_resolves_unique_capability():
    search = _tool(
        "documents.search",
        capability="document.search",
    )
    registry = StubToolRegistry([search])
    resolver = MCPCapabilityResolver(registry)

    resolved = await resolver.resolve_tools_for_agent(
        ["document.search"],
    )

    assert resolved == [search]


@pytest.mark.asyncio
async def test_exact_tool_name_takes_precedence_over_capability_match():
    exact = _tool(
        "document.search",
        capability="other.capability",
    )
    capability_match = _tool(
        "other.search",
        capability="document.search",
    )
    registry = StubToolRegistry([exact, capability_match])
    resolver = MCPCapabilityResolver(registry)

    resolved = await resolver.resolve_tools_for_agent(
        ["document.search"],
    )

    assert resolved == [exact]


@pytest.mark.asyncio
async def test_unknown_tool_or_capability_raises_resolution_error():
    registry = StubToolRegistry([])
    resolver = MCPCapabilityResolver(registry)

    try:
        await resolver.resolve_tools_for_agent(["missing.tool"])
    except MCPCapabilityNotFoundError as exc:
        assert str(exc) == "No tool or capability matches 'missing.tool'."
    else:
        raise AssertionError("Expected MCPCapabilityResolutionError")


@pytest.mark.asyncio
async def test_duplicate_capability_raises_ambiguity_error():
    first = _tool(
        "documents.search",
        capability="document.search",
    )
    second = _tool(
        "knowledge.search",
        capability="document.search",
    )
    registry = StubToolRegistry([first, second])
    resolver = MCPCapabilityResolver(registry)

    try:
        await resolver.resolve_tools_for_agent(["document.search"])
    except MCPCapabilityAmbiguousError as exc:
        assert (
            str(exc) == "Capability 'document.search' is ambiguous; "
            "matched tools: documents.search, knowledge.search."
        )
    else:
        raise AssertionError("Expected MCPCapabilityResolutionError")


@pytest.mark.asyncio
async def test_duplicate_requested_names_do_not_duplicate_tool():
    search = _tool(
        "documents.search",
        capability="document.search",
    )
    registry = StubToolRegistry([search])
    resolver = MCPCapabilityResolver(registry)

    resolved = await resolver.resolve_tools_for_agent(
        ["documents.search", "document.search", "documents.search"],
    )

    assert resolved == [search]


@pytest.mark.asyncio
async def test_disabled_tool_is_not_resolved():
    disabled = _tool(
        "documents.search",
        capability="document.search",
        enabled=False,
    )
    registry = StubToolRegistry([disabled])
    resolver = MCPCapabilityResolver(registry)

    try:
        await resolver.resolve_tools_for_agent(["document.search"])
    except MCPCapabilityNotFoundError as exc:
        assert str(exc) == "No tool or capability matches 'document.search'."
    else:
        raise AssertionError("Expected MCPCapabilityResolutionError")


@pytest.mark.asyncio
async def test_resolution_preserves_requested_order():
    search = _tool(
        "documents.search",
        capability="document.search",
    )
    query = _tool(
        "vehicle.query",
        capability="vehicle.data.query",
    )
    registry = StubToolRegistry([search, query])
    resolver = MCPCapabilityResolver(registry)

    resolved = await resolver.resolve_tools_for_agent(
        ["vehicle.data.query", "documents.search"],
    )

    assert resolved == [query, search]


@pytest.mark.asyncio
async def test_empty_requested_name_is_rejected():
    registry = StubToolRegistry([])
    resolver = MCPCapabilityResolver(registry)

    try:
        await resolver.resolve_tools_for_agent([""])
    except ValueError as exc:
        assert str(exc) == "Requested tool name must not be empty."
    else:
        raise AssertionError("Expected ValueError")
