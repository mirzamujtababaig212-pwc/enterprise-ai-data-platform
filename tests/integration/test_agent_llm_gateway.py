from __future__ import annotations

import pytest

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import (
    AgentDefinition,
    AgentRequest,
    AgentResponse,
)
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.llm_gateway.routing.router import Router


class GatewayBackedAgent:
    """
    Agent implementation that uses the real LLM Gateway through
    AgentExecutionContext.
    """

    def __init__(
        self,
        definition: AgentDefinition,
    ) -> None:
        self._definition = definition

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        result = await context.llm.generate(
            prompt=context.request.input,
            provider="mock",
            user_id=context.user_id,
        )

        return AgentResponse(
            agent_name=self.definition.name,
            output=result.text,
            session_id=context.session_id,
            metadata={
                "provider": result.provider,
                "model": result.model,
            },
        )


async def build_runtime() -> AgentRuntime:
    """
    Build an AgentRuntime using the real LLM Gateway Router.

    No fake Gateway is used here. The Router uses the production
    routing stack and the deterministic MockProvider.
    """

    definition = AgentDefinition(
        name="gateway-backed-agent",
        description="Agent backed by the enterprise LLM Gateway.",
        system_prompt="You are a gateway-backed enterprise AI agent.",
        model="mock-gpt",
    )

    agent = GatewayBackedAgent(definition)

    registry = InMemoryAgentRegistry()
    await registry.register(agent)

    gateway = Router()

    return AgentRuntime(
        registry,
        llm_gateway=gateway,
    )


@pytest.mark.asyncio
async def test_agent_runtime_uses_real_llm_gateway_router() -> None:
    """
    Verify the complete Agent → LLM Gateway → routing → MockProvider path.
    """

    runtime = await build_runtime()

    response = await runtime.run(
        "gateway-backed-agent",
        AgentRequest(
            input="Explain Retrieval-Augmented Generation.",
            user_id="integration-user",
            session_id="integration-session",
        ),
    )

    assert response.agent_name == "gateway-backed-agent"
    assert response.session_id == "integration-session"

    assert response.output == ("Mock response: Explain Retrieval-Augmented Generation.")

    assert response.metadata == {
        "provider": "mock",
        "model": "mock-gpt",
    }


@pytest.mark.asyncio
async def test_agent_runtime_preserves_gateway_request_context() -> None:
    """
    Verify that request context reaches the agent and the Gateway-backed
    LLM capability without being owned by the Gateway itself.
    """

    runtime = await build_runtime()

    response = await runtime.run(
        "gateway-backed-agent",
        AgentRequest(
            input="Hello Enterprise AI.",
            user_id="user-123",
            session_id="session-123",
        ),
    )

    assert response.output == "Mock response: Hello Enterprise AI."
    assert response.session_id == "session-123"
    assert response.metadata["provider"] == "mock"
    assert response.metadata["model"] == "mock-gpt"
