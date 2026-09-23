import pytest

from tools.authorization.models import ToolAuthorizationRequest
from tools.authorization.policy import (
    CapabilityAuthorizationPolicy,
    CompositeToolAuthorizationPolicy,
    MetadataAuthorizationPolicy,
    ToolAuthorizationPolicyResult,
)


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


@pytest.mark.asyncio
async def test_composite_policy_allows_when_all_policies_allow():
    policy = CompositeToolAuthorizationPolicy(
        [
            MetadataAuthorizationPolicy(
                {"source": "mcp"},
                policy_id="provenance_policy",
                policy_version="1.0",
            ),
            CapabilityAuthorizationPolicy(
                allowed_capabilities={"document.search"},
                allowed_risk_tiers={"low"},
                allowed_side_effects={False},
                policy_id="capability_policy",
                policy_version="2.0",
            ),
        ],
        policy_id="enterprise_policy",
        policy_version="3.0",
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
        metadata={
            "source": "mcp",
            "capability": "document.search",
            "risk_tier": "low",
            "side_effect": False,
        },
    )

    result = await policy.evaluate(request)

    assert result.allowed is True
    assert result.reason == "All authorization policies satisfied."
    assert result.policy_id == "enterprise_policy"
    assert result.policy_version == "3.0"


@pytest.mark.asyncio
async def test_composite_policy_stops_at_first_denial_and_preserves_denial_metadata():
    policy = CompositeToolAuthorizationPolicy(
        [
            MetadataAuthorizationPolicy(
                {"source": "mcp"},
                policy_id="provenance_policy",
                policy_version="4.0",
            ),
            CapabilityAuthorizationPolicy(
                allowed_risk_tiers={"low"},
                policy_id="capability_policy",
                policy_version="5.0",
            ),
        ]
    )

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="delete_document",
        metadata={
            "source": "mcp",
            "risk_tier": "high",
        },
    )

    result = await policy.evaluate(request)

    assert result.allowed is False
    assert "risk_tier='high'" in result.reason
    assert result.policy_id == "capability_policy"
    assert result.policy_version == "5.0"


@pytest.mark.asyncio
async def test_composite_policy_does_not_evaluate_later_policy_after_denial():
    class RecordingPolicy:
        def __init__(self, result):
            self.result = result
            self.evaluated = False

        async def evaluate(self, request):
            self.evaluated = True
            return self.result

    first = RecordingPolicy(
        ToolAuthorizationPolicyResult(
            allowed=False,
            reason="first policy denied",
            policy_id="first",
            policy_version="1.0",
        )
    )
    second = RecordingPolicy(
        ToolAuthorizationPolicyResult(
            allowed=True,
            reason="second policy allowed",
            policy_id="second",
            policy_version="1.0",
        )
    )

    policy = CompositeToolAuthorizationPolicy([first, second])

    request = ToolAuthorizationRequest(
        principal="agent:research",
        tool_name="search_documents",
    )

    result = await policy.evaluate(request)

    assert result.allowed is False
    assert result.reason == "first policy denied"
    assert result.policy_id == "first"
    assert result.policy_version == "1.0"
    assert first.evaluated is True
    assert second.evaluated is False


def test_composite_policy_rejects_empty_policy_list():
    with pytest.raises(
        ValueError,
        match="At least one authorization policy is required",
    ):
        CompositeToolAuthorizationPolicy([])


def test_composite_policy_rejects_empty_policy_identity():
    policy = MetadataAuthorizationPolicy({})

    with pytest.raises(
        ValueError,
        match="policy_id must not be empty",
    ):
        CompositeToolAuthorizationPolicy(
            [policy],
            policy_id="",
        )

    with pytest.raises(
        ValueError,
        match="policy_version must not be empty",
    ):
        CompositeToolAuthorizationPolicy(
            [policy],
            policy_version="",
        )
