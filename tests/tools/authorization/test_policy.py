import pytest

from tools.authorization.models import ToolAuthorizationRequest
from tools.authorization.policy import MetadataAuthorizationPolicy


@pytest.mark.asyncio
async def test_metadata_policy_allows_matching_metadata():
    policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "document-server",
        }
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
        metadata={
            "source": "mcp",
            "mcp_server": "document-server",
        },
    )

    result = await policy.evaluate(request)

    assert result.allowed is True
    assert result.reason == "Authorization metadata requirements satisfied."
    assert result.policy_id == "metadata_policy"
    assert result.policy_version == "1.0"


@pytest.mark.asyncio
async def test_metadata_policy_denies_missing_metadata():
    policy = MetadataAuthorizationPolicy(
        {
            "mcp_server": "document-server",
        }
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
    )

    result = await policy.evaluate(request)

    assert result.allowed is False
    assert "mcp_server='document-server'" in result.reason
    assert result.policy_id == "metadata_policy"
    assert result.policy_version == "1.0"


@pytest.mark.asyncio
async def test_metadata_policy_denies_mismatched_metadata():
    policy = MetadataAuthorizationPolicy(
        {
            "mcp_server": "document-server",
        }
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
        metadata={
            "mcp_server": "finance-server",
        },
    )

    result = await policy.evaluate(request)

    assert result.allowed is False


@pytest.mark.asyncio
async def test_metadata_policy_allows_empty_requirements():
    policy = MetadataAuthorizationPolicy({})

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
    )

    result = await policy.evaluate(request)

    assert result.allowed is True


@pytest.mark.asyncio
async def test_capability_policy_allows_matching_capability_metadata():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        allowed_capabilities={"document.search"},
        allowed_risk_tiers={"low"},
        allowed_side_effects={False},
        required_permission_scope="documents:read",
        policy_id="enterprise_capability_policy",
        policy_version="2.0",
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
        metadata={
            "capability": "document.search",
            "risk_tier": "low",
            "side_effect": False,
            "permission_scope": "documents:read",
        },
    )

    result = await policy.evaluate(request)

    assert result.allowed is True
    assert result.reason == "Authorization capability requirements satisfied."
    assert result.policy_id == "enterprise_capability_policy"
    assert result.policy_version == "2.0"


@pytest.mark.asyncio
async def test_capability_policy_denies_disallowed_risk_tier():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        allowed_capabilities={"document.search"},
        allowed_risk_tiers={"low"},
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
        metadata={
            "capability": "document.search",
            "risk_tier": "high",
        },
    )

    result = await policy.evaluate(request)

    assert result.allowed is False
    assert "risk_tier='high'" in result.reason
    assert result.policy_id == "capability_policy"
    assert result.policy_version == "1.0"


@pytest.mark.asyncio
async def test_capability_policy_denies_disallowed_side_effect():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        allowed_side_effects={False},
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="delete_document",
        metadata={
            "side_effect": True,
        },
    )

    result = await policy.evaluate(request)

    assert result.allowed is False
    assert "side_effect=True" in result.reason


@pytest.mark.asyncio
async def test_capability_policy_denies_permission_scope_mismatch():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        required_permission_scope="documents:read",
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
        metadata={
            "permission_scope": "documents:write",
        },
    )

    result = await policy.evaluate(request)

    assert result.allowed is False
    assert "permission_scope='documents:write'" in result.reason


@pytest.mark.asyncio
async def test_capability_policy_denies_missing_capability():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        allowed_capabilities={"document.search"},
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
        metadata={},
    )

    result = await policy.evaluate(request)

    assert result.allowed is False
    assert "capability=None" in result.reason
