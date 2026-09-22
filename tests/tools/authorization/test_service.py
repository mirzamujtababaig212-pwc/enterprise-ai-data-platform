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
