from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import AgentDefinition, AgentRequest, AgentResponse
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.agents.tool_calls import AgentToolCall
from rag.governance import GovernancePolicy
from tools.authorization.in_memory import InMemoryToolAuthorizer
from tools.authorization.policy import (
    CapabilityAuthorizationPolicy,
    MetadataAuthorizationPolicy,
)
from tools.authorization.service import ToolAuthorizationService
from tools.execution.service import ToolExecutionService
from tools.mcp.config import MCPServerConfig, MCPToolCapability
from tools.mcp.manager import MCPServerManager
from tools.mcp.recovery import MCPRecoveryPolicy
from tools.registry.in_memory import InMemoryToolRegistry

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tools" / "mcp" / "fixtures"
SEARCH_SERVER = FIXTURES_DIR / "test_server.py"


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


def make_agent() -> MCPCallingAgent:
    return MCPCallingAgent(
        AgentDefinition(
            name="governed-mcp-agent-test",
            description="Agent integration test for governed MCP authorization.",
            system_prompt=(
                "You are an enterprise AI agent. "
                "Use the search_documents MCP tool when required."
            ),
            model="mock-gpt",
            tool_names=("search_documents",),
        )
    )


def make_governance_policy() -> GovernancePolicy:
    return GovernancePolicy(
        required_metadata={
            "classification": "internal",
            "tenant": "deldai",
        }
    )


@pytest.mark.asyncio
async def test_agent_runtime_authorizes_real_mcp_tool_before_execution() -> None:
    registry = InMemoryToolRegistry()

    authorization_policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "document-server",
        }
    )
    authorizer = InMemoryToolAuthorizer(
        policy=authorization_policy,
    )
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

    agent = make_agent()
    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
    )

    await authorizer.allow(
        "user-mcp-456",
        "search_documents",
    )

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

        response = await runtime.run(
            "governed-mcp-agent-test",
            AgentRequest(
                input="Find enterprise AI architecture information.",
                session_id="session-mcp-123",
                user_id="user-mcp-456",
                principal="user-mcp-456",
                governance_policy=make_governance_policy(),
                metadata={
                    "source": "agent-mcp-authorization-test",
                },
            ),
        )

        assert response.agent_name == "governed-mcp-agent-test"
        assert response.session_id == "session-mcp-123"

        assert agent.last_tool_results is not None
        assert len(agent.last_tool_results) == 1

        tool_result = agent.last_tool_results[0]

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
        await manager.disconnect_all()


@pytest.mark.asyncio
async def test_recovered_mcp_tool_remains_governed_and_executable() -> None:
    class FailureInjectingMCPClient:
        def __init__(self, delegate) -> None:
            self._delegate = delegate
            self.fail_next_call = False

        async def connect(self) -> None:
            await self._delegate.connect()

        async def disconnect(self) -> None:
            await self._delegate.disconnect()

        async def send_ping(self) -> None:
            await self._delegate.send_ping()

        async def list_tools(self):
            return await self._delegate.list_tools()

        async def call_tool(
            self,
            name: str,
            arguments: dict,
            *,
            meta: dict | None = None,
        ):
            if self.fail_next_call:
                self.fail_next_call = False
                raise RuntimeError("simulated MCP transport failure")

            return await self._delegate.call_tool(
                name,
                arguments,
                meta=meta,
            )

    registry = InMemoryToolRegistry()

    authorization_policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "document-server",
        }
    )

    original_evaluate = authorization_policy.evaluate
    authorization_calls = []

    async def recording_evaluate(request):
        authorization_calls.append(request)
        return await original_evaluate(request)

    authorization_policy.evaluate = recording_evaluate

    authorizer = InMemoryToolAuthorizer(
        policy=authorization_policy,
    )
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
        recovery_policy=MCPRecoveryPolicy(
            max_attempts=2,
            initial_backoff=0.0,
            max_backoff=0.0,
            cooldown=0.0,
        ),
    )

    await manager.register_server(config)

    real_client = await manager.get_client("document-server")
    failure_client = FailureInjectingMCPClient(real_client)
    manager._servers["document-server"].client = failure_client

    await authorizer.allow(
        "user-mcp-recovery-789",
        "search_documents",
    )

    try:
        definitions = await manager.connect_and_discover("document-server")

        assert definitions[0].metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        registered_before = await registry.get("search_documents")
        assert registered_before is not None

        first_result = await execution_service.execute(
            "search_documents",
            {"query": "enterprise AI"},
            principal="user-mcp-recovery-789",
        )

        assert first_result.success is True
        assert first_result.output == {
            "query": "enterprise AI",
            "results": [
                {
                    "id": "document-1",
                    "content": "Enterprise AI platform architecture.",
                }
            ],
        }

        assert len(authorization_calls) == 1
        assert authorization_calls[0].metadata == definitions[0].metadata

        failure_client.fail_next_call = True

        with_failure = await execution_service.execute(
            "search_documents",
            {"query": "enterprise AI"},
            principal="user-mcp-recovery-789",
        )

        assert with_failure.success is False
        assert "simulated MCP transport failure" in with_failure.error

        assert len(authorization_calls) == 2
        assert authorization_calls[1].metadata == definitions[0].metadata

        recovered_definitions = await manager.recover_server("document-server")

        assert manager.is_connected("document-server") is True
        assert recovered_definitions == [
            definitions[0],
        ]

        registered_after = await registry.get("search_documents")
        assert registered_after is not None
        assert registered_after.definition.metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        recovered_result = await execution_service.execute(
            "search_documents",
            {"query": "enterprise AI"},
            principal="user-mcp-recovery-789",
        )

        assert recovered_result.success is True
        assert recovered_result.output == {
            "query": "enterprise AI",
            "results": [
                {
                    "id": "document-1",
                    "content": "Enterprise AI platform architecture.",
                }
            ],
        }

        assert len(authorization_calls) == 3
        assert authorization_calls[2].metadata == registered_after.definition.metadata

    finally:
        await manager.disconnect_all()


@pytest.mark.asyncio
async def test_agent_runtime_denies_real_mcp_tool_before_server_execution() -> None:
    registry = InMemoryToolRegistry()

    authorization_policy = MetadataAuthorizationPolicy(
        {
            "source": "mcp",
            "mcp_server": "finance-server",
        }
    )
    authorizer = InMemoryToolAuthorizer(
        policy=authorization_policy,
    )
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

    agent = make_agent()
    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
    )

    await authorizer.allow(
        "user-mcp-456",
        "search_documents",
    )

    try:
        definitions = await manager.connect_and_discover("document-server")

        assert definitions[0].metadata == {
            "source": "mcp",
            "mcp_server": "document-server",
            "capability": "unclassified",
            "risk_tier": "unknown",
            "side_effect": True,
        }

        response = await runtime.run(
            "governed-mcp-agent-test",
            AgentRequest(
                input="Find enterprise AI architecture information.",
                session_id="session-mcp-denied",
                user_id="user-mcp-456",
                principal="user-mcp-456",
                governance_policy=make_governance_policy(),
                metadata={
                    "source": "agent-mcp-authorization-test",
                },
            ),
        )

        assert response.agent_name == "governed-mcp-agent-test"

        assert agent.last_tool_results is not None
        assert len(agent.last_tool_results) == 1

        tool_result = agent.last_tool_results[0]

        assert tool_result.tool_name == "search_documents"
        assert tool_result.success is False
        assert tool_result.output is None
        assert "mcp_server='finance-server'" in tool_result.error

    finally:
        await manager.disconnect_all()


@pytest.mark.asyncio
async def test_agent_runtime_resolves_and_authorizes_real_mcp_capability() -> None:
    registry = InMemoryToolRegistry()

    authorization_policy = CapabilityAuthorizationPolicy(
        allowed_capabilities={"document.search"},
        allowed_risk_tiers={"low"},
        allowed_side_effects={False},
        required_permission_scope="documents:read",
        policy_id="enterprise_capability_policy",
        policy_version="2.0",
    )

    authorizer = InMemoryToolAuthorizer(
        policy=authorization_policy,
    )
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
        tool_capabilities={
            "search_documents": MCPToolCapability(
                capability="document.search",
                risk_tier="low",
                side_effect=False,
                permission_scope="documents:read",
            ),
        },
    )

    await manager.register_server(config)

    agent = MCPCallingAgent(
        AgentDefinition(
            name="governed-mcp-capability-agent-test",
            description=("Agent integration test for capability-based MCP authorization."),
            system_prompt=(
                "You are an enterprise AI agent. "
                "Use the document.search capability when required."
            ),
            model="mock-gpt",
            tool_names=("document.search",),
        )
    )

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(agent)

    runtime = AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
    )

    await authorizer.allow(
        "user-mcp-capability-456",
        "search_documents",
    )

    try:
        definitions = await manager.connect_and_discover("document-server")

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

        response = await runtime.run(
            "governed-mcp-capability-agent-test",
            AgentRequest(
                input="Find enterprise AI architecture information.",
                session_id="session-mcp-capability-auth-123",
                user_id="user-mcp-capability-456",
                principal="user-mcp-capability-456",
                governance_policy=make_governance_policy(),
                metadata={
                    "source": "agent-mcp-capability-authorization-test",
                },
            ),
        )

        assert response.agent_name == "governed-mcp-capability-agent-test"
        assert response.session_id == "session-mcp-capability-auth-123"

        assert agent.last_tool_results is not None
        assert len(agent.last_tool_results) == 1

        tool_result = agent.last_tool_results[0]

        # The agent declares the capability, while the execution path
        # resolves it to the provider-specific MCP tool name.
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
        await manager.disconnect_all()
