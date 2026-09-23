from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.authorization.in_memory import InMemoryToolAuthorizer
from tools.authorization.policy import MetadataAuthorizationPolicy
from tools.authorization.service import ToolAuthorizationService
from tools.execution.service import ToolExecutionService
from tools.mcp.config import MCPServerConfig
from tools.mcp.manager import MCPServerManager
from tools.registry.in_memory import InMemoryToolRegistry
from tools.execution.idempotency import InMemoryToolExecutionIdempotencyStore
from tools.execution.context import ToolExecutionContext

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tools" / "mcp" / "fixtures"

SEARCH_SERVER = FIXTURES_DIR / "test_server.py"


@pytest.mark.asyncio
async def test_real_mcp_tool_metadata_reaches_authorization_policy() -> None:
    registry = InMemoryToolRegistry()

    policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "document-server",
        }
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)
    authorization_service = ToolAuthorizationService(authorizer)

    execution_service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    manager = MCPServerManager(registry)

    config = MCPServerConfig(
        name="document-server",
        transport="stdio",
        command=sys.executable,
        args=(str(SEARCH_SERVER),),
    )

    await manager.register_server(config)

    try:
        definitions = await manager.connect_and_discover("document-server")

        assert [definition.name for definition in definitions] == [
            "search_documents",
        ]

        assert definitions[0].metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        discovered_tool = await registry.get("search_documents")

        assert discovered_tool is not None
        assert discovered_tool.definition.metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        await authorizer.allow(
            "agent:research",
            "search_documents",
        )

        allowed_result = await execution_service.execute(
            "search_documents",
            {"query": "enterprise AI"},
            principal="agent:research",
        )

        assert allowed_result.success is True
        assert allowed_result.output == {
            "query": "enterprise AI",
            "results": [
                {
                    "id": "document-1",
                    "content": "Enterprise AI platform architecture.",
                }
            ],
        }

    finally:
        await manager.disconnect_all()


@pytest.mark.asyncio
async def test_real_mcp_tool_is_denied_when_authorization_metadata_policy_does_not_match() -> None:
    registry = InMemoryToolRegistry()

    policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "finance-server",
        }
    )
    authorizer = InMemoryToolAuthorizer(policy=policy)
    authorization_service = ToolAuthorizationService(authorizer)

    execution_service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    manager = MCPServerManager(registry)

    config = MCPServerConfig(
        name="document-server",
        transport="stdio",
        command=sys.executable,
        args=(str(SEARCH_SERVER),),
    )

    await manager.register_server(config)

    try:
        definitions = await manager.connect_and_discover("document-server")

        assert definitions[0].metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        await authorizer.allow(
            "agent:research",
            "search_documents",
        )

        denied_result = await execution_service.execute(
            "search_documents",
            {"query": "enterprise AI"},
            principal="agent:research",
        )

        assert denied_result.success is False
        assert "mcp_server='finance-server'" in denied_result.error

    finally:
        await manager.disconnect_all()


@pytest.mark.asyncio
async def test_real_mcp_tool_replays_completed_idempotent_result() -> None:
    registry = InMemoryToolRegistry()
    idempotency_store = InMemoryToolExecutionIdempotencyStore()

    execution_service = ToolExecutionService(
        registry,
        idempotency_store=idempotency_store,
    )

    manager = MCPServerManager(registry)

    config = MCPServerConfig(
        name="document-server",
        transport="stdio",
        command=sys.executable,
        args=(str(SEARCH_SERVER),),
    )

    await manager.register_server(config)

    try:
        definitions = await manager.connect_and_discover("document-server")

        assert [definition.name for definition in definitions] == [
            "search_documents",
        ]

        context = ToolExecutionContext(
            run_id="run-mcp-idempotency",
            call_id="call-search-documents",
        )

        first = await execution_service.execute(
            "search_documents",
            {"query": "enterprise AI"},
            execution_context=context,
        )

        second = await execution_service.execute(
            "search_documents",
            {"query": "enterprise AI"},
            execution_context=context,
        )

        assert first.success is True
        assert second.success is True
        assert second == first

        assert first.output == {
            "query": "enterprise AI",
            "results": [
                {
                    "id": "document-1",
                    "content": "Enterprise AI platform architecture.",
                }
            ],
        }

    finally:
        await manager.disconnect_all()
