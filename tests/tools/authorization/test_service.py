import pytest

from tools.authorization.in_memory import (
    InMemoryToolAuthorizer,
)
from tools.authorization.models import (
    ToolAuthorizationRequest,
    ToolAuthorizationResult,
)
from tools.authorization.policy import MetadataAuthorizationPolicy
from tools.authorization.service import (
    ToolAuthorizationService,
)


class RecordingToolAuthorizer:
    def __init__(self) -> None:
        self.requests = []

    async def authorize(self, request):
        self.requests.append(request)
        return ToolAuthorizationResult(
            principal=request.principal,
            tool_name=request.tool_name,
            allowed=True,
        )


@pytest.mark.asyncio
async def test_authorized_tool_is_allowed():
    authorizer = InMemoryToolAuthorizer()

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "search_documents",
    )

    assert result.allowed is True
    assert result.principal == "agent:research"
    assert result.tool_name == "search_documents"
    assert result.reason == "Tool is authorized."


@pytest.mark.asyncio
async def test_unauthorized_tool_is_denied():
    authorizer = InMemoryToolAuthorizer()

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "delete_database",
    )

    assert result.allowed is False
    assert result.principal == "agent:research"
    assert result.tool_name == "delete_database"
    assert result.reason == "Tool is not authorized for this principal."


@pytest.mark.asyncio
async def test_deny_removes_existing_permission():
    authorizer = InMemoryToolAuthorizer()

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    await authorizer.deny(
        "agent:research",
        "search_documents",
    )

    result = await authorizer.authorize(
        ToolAuthorizationRequest(
            principal="agent:research",
            tool_name="search_documents",
        )
    )

    assert result.allowed is False


@pytest.mark.asyncio
async def test_multiple_tools_can_be_authorized():
    authorizer = InMemoryToolAuthorizer()

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    await authorizer.allow(
        "agent:research",
        "summarize_documents",
    )

    service = ToolAuthorizationService(authorizer)

    search_result = await service.authorize(
        "agent:research",
        "search_documents",
    )

    summarize_result = await service.authorize(
        "agent:research",
        "summarize_documents",
    )

    assert search_result.allowed is True
    assert summarize_result.allowed is True


@pytest.mark.asyncio
async def test_permissions_are_principal_specific():
    authorizer = InMemoryToolAuthorizer()

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    service = ToolAuthorizationService(authorizer)

    authorized = await service.authorize(
        "agent:research",
        "search_documents",
    )

    unauthorized = await service.authorize(
        "agent:finance",
        "search_documents",
    )

    assert authorized.allowed is True
    assert unauthorized.allowed is False


@pytest.mark.asyncio
async def test_empty_principal_is_rejected():
    authorizer = InMemoryToolAuthorizer()

    service = ToolAuthorizationService(authorizer)

    with pytest.raises(
        ValueError,
        match="Principal must not be empty",
    ):
        await service.authorize(
            "",
            "search_documents",
        )


@pytest.mark.asyncio
async def test_empty_tool_name_is_rejected():
    authorizer = InMemoryToolAuthorizer()

    service = ToolAuthorizationService(authorizer)

    with pytest.raises(
        ValueError,
        match="Tool name must not be empty",
    ):
        await service.authorize(
            "agent:research",
            "",
        )


@pytest.mark.asyncio
async def test_allow_rejects_empty_principal():
    authorizer = InMemoryToolAuthorizer()

    with pytest.raises(
        ValueError,
        match="Principal must not be empty",
    ):
        await authorizer.allow(
            "",
            "search_documents",
        )


@pytest.mark.asyncio
async def test_allow_rejects_empty_tool_name():
    authorizer = InMemoryToolAuthorizer()

    with pytest.raises(
        ValueError,
        match="Tool name must not be empty",
    ):
        await authorizer.allow(
            "agent:research",
            "",
        )


@pytest.mark.asyncio
async def test_authorization_service_passes_metadata_to_authorizer():
    authorizer = RecordingToolAuthorizer()
    service = ToolAuthorizationService(authorizer)

    metadata = {
        "source": "mcp",
        "mcp_server": "document-server",
    }

    result = await service.authorize(
        "agent:research",
        "search_documents",
        metadata=metadata,
    )

    assert result.allowed is True
    assert len(authorizer.requests) == 1
    assert authorizer.requests[0].metadata == metadata


@pytest.mark.asyncio
async def test_authorizer_allows_tool_when_permission_and_policy_match():
    policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "document-server",
        }
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "search_documents",
        metadata={
            "source": "mcp",
            "mcp_server": "document-server",
        },
    )

    assert result.allowed is True
    assert result.reason == "Tool is authorized."
    assert result.policy_id == "metadata_policy"
    assert result.policy_version == "1.0"


@pytest.mark.asyncio
async def test_authorizer_denies_tool_when_metadata_policy_fails():
    policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "document-server",
        }
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "search_documents",
        metadata={
            "source": "mcp",
            "mcp_server": "finance-server",
        },
    )

    assert result.allowed is False
    assert "mcp_server='document-server'" in result.reason
    assert result.policy_id == "metadata_policy"
    assert result.policy_version == "1.0"


@pytest.mark.asyncio
async def test_authorizer_denies_without_tool_permission_even_when_policy_matches():
    policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "document-server",
        }
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "search_documents",
        metadata={
            "source": "mcp",
            "mcp_server": "document-server",
        },
    )

    assert result.allowed is False
    assert result.reason == "Tool is not authorized for this principal."
    assert result.policy_id is None
    assert result.policy_version is None


@pytest.mark.asyncio
async def test_authorizer_without_policy_preserves_existing_behavior():
    authorizer = InMemoryToolAuthorizer()

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "search_documents",
        metadata={
            "source": "mcp",
            "mcp_server": "untrusted-server",
        },
    )

    assert result.allowed is True
    assert result.reason == "Tool is authorized."


@pytest.mark.asyncio
async def test_authorizer_allows_tool_when_permission_and_capability_policy_match():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        allowed_capabilities={"document.search"},
        allowed_risk_tiers={"low"},
        allowed_side_effects={False},
        required_permission_scope="documents:read",
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "search_documents",
        metadata={
            "capability": "document.search",
            "risk_tier": "low",
            "side_effect": False,
            "permission_scope": "documents:read",
        },
    )

    assert result.allowed is True
    assert result.policy_id == "capability_policy"
    assert result.policy_version == "1.0"


@pytest.mark.asyncio
async def test_authorizer_denies_tool_when_capability_policy_rejects_risk():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        allowed_capabilities={"document.search"},
        allowed_risk_tiers={"low"},
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "search_documents",
        metadata={
            "capability": "document.search",
            "risk_tier": "high",
        },
    )

    assert result.allowed is False
    assert "risk_tier='high'" in result.reason
    assert result.policy_id == "capability_policy"
    assert result.policy_version == "1.0"


@pytest.mark.asyncio
async def test_authorizer_denies_side_effecting_tool_when_policy_disallows_it():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        allowed_side_effects={False},
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)

    await authorizer.allow(
        "agent:research",
        "delete_document",
    )

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "delete_document",
        metadata={
            "side_effect": True,
        },
    )

    assert result.allowed is False
    assert "side_effect=True" in result.reason
    assert result.policy_id == "capability_policy"


@pytest.mark.asyncio
async def test_authorizer_denies_permission_scope_mismatch():
    from tools.authorization.policy import CapabilityAuthorizationPolicy

    policy = CapabilityAuthorizationPolicy(
        required_permission_scope="documents:read",
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)

    await authorizer.allow(
        "agent:research",
        "search_documents",
    )

    service = ToolAuthorizationService(authorizer)

    result = await service.authorize(
        "agent:research",
        "search_documents",
        metadata={
            "permission_scope": "documents:write",
        },
    )

    assert result.allowed is False
    assert "permission_scope='documents:write'" in result.reason
    assert result.policy_id == "capability_policy"
