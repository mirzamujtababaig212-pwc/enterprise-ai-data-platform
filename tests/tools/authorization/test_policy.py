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
