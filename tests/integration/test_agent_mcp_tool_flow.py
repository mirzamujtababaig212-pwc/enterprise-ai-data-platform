from __future__ import annotations

from pathlib import Path
import sys

import pytest
from mcp import StdioServerParameters

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import AgentDefinition, AgentRequest, AgentResponse
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.agents.tool_calls import AgentToolCall
from rag.governance import GovernancePolicy
from tools.mcp.config import MCPToolCapability
from tools.mcp.discovery import MCPToolDiscoveryService
from tools.mcp.sdk_client import MCPPythonSDKClient
from tools.registry.in_memory import InMemoryToolRegistry

SERVER_PATH = Path(__file__).resolve().parents[1] / "tools" / "mcp" / "fixtures" / "test_server.py"


class MCPCallingAgent:
    def __init__(self, definition: AgentDefinition) -> None:
        self._definition = definition
        self.last_tool_results = None

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        self.last_tool_results = await context.execute_tool_calls(
            (
                AgentToolCall(
                    call_id="call-1",
                    name="search_documents",
                    arguments={
                        "query": "enterprise AI",
                    },
                ),
            )
        )

        return AgentResponse(
            agent_name=self.definition.name,
            output=str(self.last_tool_results),
            session_id=context.session_id,
        )


@pytest.mark.asyncio
async def test_agent_runtime_executes_real_mcp_tool_with_platform_context() -> None:
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            str(SERVER_PATH),
        ],
    )

    client = MCPPythonSDKClient(server_parameters)
    tool_registry = InMemoryToolRegistry()

    try:
        await client.connect()

        discovery = MCPToolDiscoveryService(
            client=client,
            registry=tool_registry,
        )

        definitions = await discovery.discover_and_register()

        assert [definition.name for definition in definitions] == [
            "search_documents",
        ]

        agent_definition = AgentDefinition(
            name="mcp-enabled-agent-test",
            description="Agent integration test for a real MCP tool.",
            system_prompt=(
                "You are an enterprise AI agent. "
                "Use the search_documents MCP tool when required."
            ),
            model="mock-gpt",
            tool_names=("search_documents",),
        )

        agent = MCPCallingAgent(agent_definition)

        agent_registry = InMemoryAgentRegistry()
        await agent_registry.register(agent)

        runtime = AgentRuntime(
            agent_registry,
            tool_registry=tool_registry,
        )

        governance_policy = GovernancePolicy(
            required_metadata={
                "classification": "internal",
                "tenant": "deldai",
            }
        )

        response = await runtime.run(
            "mcp-enabled-agent-test",
            AgentRequest(
                input="Find enterprise AI architecture information.",
                session_id="session-mcp-123",
                user_id="user-mcp-456",
                governance_policy=governance_policy,
                metadata={
                    "source": "agent-mcp-integration-test",
                    "request_type": "search",
                },
            ),
        )

        assert response.agent_name == "mcp-enabled-agent-test"
        assert response.session_id == "session-mcp-123"

        assert agent.last_tool_results is not None
        assert len(agent.last_tool_results) == 1

        tool_result = agent.last_tool_results[0]

        assert tool_result.tool_name == "search_documents"
        assert tool_result.success is True

        assert tool_result.output["query"] == "enterprise AI"
        assert tool_result.output["results"] == [
            {
                "id": "document-1",
                "content": "Enterprise AI platform architecture.",
            }
        ]

        # The MCP server receives only its declared tool arguments.
        # Platform execution context remains inside DELDAI's execution boundary.
        assert "governance_policy" not in tool_result.output
        assert "user_id" not in tool_result.output
        assert "session_id" not in tool_result.output
        assert "request_metadata" not in tool_result.output

    finally:
        await client.disconnect()


@pytest.mark.asyncio
async def test_agent_runtime_resolves_mcp_capability_to_real_tool() -> None:
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=[
            str(SERVER_PATH),
        ],
    )

    client = MCPPythonSDKClient(server_parameters)
    tool_registry = InMemoryToolRegistry()

    try:
        await client.connect()

        discovery = MCPToolDiscoveryService(
            client=client,
            registry=tool_registry,
            server_name="document-server",
            tool_capabilities={
                "search_documents": MCPToolCapability(
                    capability="document.search",
                    risk_tier="low",
                    side_effect=False,
                    permission_scope="documents:read",
                ),
            },
        )

        definitions = await discovery.discover_and_register()

        assert [definition.name for definition in definitions] == [
            "search_documents",
        ]

        assert definitions[0].metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "document.search",
            "risk_tier": "low",
            "side_effect": False,
            "permission_scope": "documents:read",
        }

        # The agent declares the Deldai capability rather than the
        # provider-specific MCP tool name.
        agent_definition = AgentDefinition(
            name="mcp-capability-agent-test",
            description="Agent integration test for MCP capability resolution.",
            system_prompt=(
                "You are an enterprise AI agent. "
                "Use the document.search capability when required."
            ),
            model="mock-gpt",
            tool_names=("document.search",),
        )

        agent = MCPCallingAgent(agent_definition)

        agent_registry = InMemoryAgentRegistry()
        await agent_registry.register(agent)

        runtime = AgentRuntime(
            agent_registry,
            tool_registry=tool_registry,
        )

        response = await runtime.run(
            "mcp-capability-agent-test",
            AgentRequest(
                input="Find enterprise AI architecture information.",
                session_id="session-mcp-capability-123",
                user_id="user-mcp-capability-456",
                metadata={
                    "source": "agent-mcp-capability-integration-test",
                    "request_type": "search",
                },
            ),
        )

        assert response.agent_name == "mcp-capability-agent-test"
        assert response.session_id == "session-mcp-capability-123"

        assert agent.last_tool_results is not None
        assert len(agent.last_tool_results) == 1

        tool_result = agent.last_tool_results[0]

        # The agent's capability declaration resolves to the concrete
        # provider tool before execution.
        assert tool_result.tool_name == "search_documents"
        assert tool_result.success is True

        assert tool_result.output == {
            "query": "enterprise AI",
            "results": [
                {
                    "id": "document-1",
                    "content": "Enterprise AI platform architecture.",
                }
            ],
        }

    finally:
        await client.disconnect()
