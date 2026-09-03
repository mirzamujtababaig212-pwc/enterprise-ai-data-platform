import pytest

from tools.authorization.in_memory import (
    InMemoryToolAuthorizer,
)
from tools.authorization.models import (
    ToolAuthorizationRequest,
)
from tools.authorization.service import (
    ToolAuthorizationService,
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
