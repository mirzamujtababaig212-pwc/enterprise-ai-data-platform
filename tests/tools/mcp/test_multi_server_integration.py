from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tools.mcp.config import MCPServerConfig
from tools.mcp.manager import MCPServerManager
from tools.registry.in_memory import InMemoryToolRegistry


FIXTURES_DIR = Path(__file__).parent / "fixtures"

SEARCH_SERVER = FIXTURES_DIR / "test_server.py"
POLICY_SERVER = FIXTURES_DIR / "policy_server.py"


@pytest.mark.asyncio
async def test_manager_manages_multiple_real_stdio_servers():
    registry = InMemoryToolRegistry()
    manager = MCPServerManager(registry)

    search_config = MCPServerConfig(
        name="document-server",
        transport="stdio",
        command=sys.executable,
        args=(str(SEARCH_SERVER),),
    )

    policy_config = MCPServerConfig(
        name="policy-server",
        transport="stdio",
        command=sys.executable,
        args=(str(POLICY_SERVER),),
    )

    await manager.register_server(search_config)
    await manager.register_server(policy_config)

    assert manager.list_servers() == [
        "document-server",
        "policy-server",
    ]

    assert manager.is_connected("document-server") is False
    assert manager.is_connected("policy-server") is False

    try:
        document_tools = await manager.connect_and_discover("document-server")

        policy_tools = await manager.connect_and_discover("policy-server")

        assert manager.is_connected("document-server") is True
        assert manager.is_connected("policy-server") is True

        assert len(document_tools) == 1
        assert document_tools[0].name == "search_documents"

        assert len(policy_tools) == 1
        assert policy_tools[0].name == "get_policy"

        registered_tools = await registry.list_tools()

        registered_names = {definition.name for definition in registered_tools}

        assert "search_documents" in registered_names
        assert "get_policy" in registered_names

        search_tool = await registry.get("search_documents")

        policy_tool = await registry.get("get_policy")

        assert search_tool is not None
        assert policy_tool is not None

        search_result = await search_tool.execute({"query": "enterprise AI"})

        assert search_result["query"] == "enterprise AI"
        assert search_result["results"] == [
            {
                "id": "document-1",
                "content": "Enterprise AI platform architecture.",
            }
        ]

        policy_result = await policy_tool.execute({"policy_name": "security"})

        assert policy_result["policy_name"] == "security"
        assert policy_result["found"] is True
        assert policy_result["policy"]["require_mfa"] is True
        assert policy_result["policy"]["audit_logging"] is True

    finally:
        await manager.disconnect_all()

    assert manager.is_connected("document-server") is False
    assert manager.is_connected("policy-server") is False
