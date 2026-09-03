from __future__ import annotations

from typing import Any

import pytest

from ai_platform.agents.llm_context import LLMGateway
from ai_platform.agents.models import (
    AgentDefinition,
    AgentRequest,
    AgentResponse,
)
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime


class FakeGateway:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        return {
            "provider": request.get("provider", "mock"),
            "model": request["model"],
            "reply": f"Generated: {request['prompt']}",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        }


class LLMUsingAgent:
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
        context,
    ) -> AgentResponse:
        result = await context.llm.generate(
            prompt=context.request.input,
            user_id=context.user_id,
        )

        return AgentResponse(
            agent_name=self.definition.name,
            output=result.text,
            session_id=context.session_id,
            metadata={
                "provider": result.provider,
                "model": result.model,
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                },
            },
        )


@pytest.mark.asyncio
async def test_runtime_exposes_llm_gateway_to_agent() -> None:
    gateway = FakeGateway()

    definition = AgentDefinition(
        name="llm-agent",
        description="LLM test agent",
        system_prompt="You are an LLM test agent.",
        model="mock-gpt",
    )

    agent = LLMUsingAgent(definition)

    registry = InMemoryAgentRegistry()
    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
        llm_gateway=gateway,
    )

    response = await runtime.run(
        "llm-agent",
        AgentRequest(
            input="Explain RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
    )

    assert response.agent_name == "llm-agent"
    assert response.output == "Generated: Explain RAG."
    assert response.session_id == "session-456"
    assert response.metadata == {
        "provider": "mock",
        "model": "mock-gpt",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }
    assert gateway.requests == [
        {
            "prompt": "Explain RAG.",
            "model": "mock-gpt",
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
            "user_id": "user-123",
        }
    ]


@pytest.mark.asyncio
async def test_runtime_uses_agent_definition_model() -> None:
    gateway = FakeGateway()

    definition = AgentDefinition(
        name="custom-model-agent",
        description="Custom model agent",
        system_prompt="You use a custom model.",
        model="custom-model",
    )

    agent = LLMUsingAgent(definition)

    registry = InMemoryAgentRegistry()
    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
        llm_gateway=gateway,
    )

    await runtime.run(
        "custom-model-agent",
        AgentRequest(input="Hello."),
    )

    assert gateway.requests[0]["model"] == "custom-model"


@pytest.mark.asyncio
async def test_runtime_rejects_llm_use_without_gateway() -> None:
    definition = AgentDefinition(
        name="llm-agent",
        description="LLM test agent",
        system_prompt="You are an LLM test agent.",
        model="mock-gpt",
    )

    agent = LLMUsingAgent(definition)

    registry = InMemoryAgentRegistry()
    await registry.register(agent)

    runtime = AgentRuntime(registry)

    with pytest.raises(
        RuntimeError,
        match="LLM Gateway is not configured for AgentRuntime",
    ):
        await runtime.run(
            "llm-agent",
            AgentRequest(input="Hello."),
        )


@pytest.mark.asyncio
async def test_runtime_passes_user_and_session_context_to_agent() -> None:
    gateway = FakeGateway()

    definition = AgentDefinition(
        name="context-agent",
        description="Context test agent",
        system_prompt="You are a context test agent.",
        model="mock-gpt",
    )

    agent = LLMUsingAgent(definition)

    registry = InMemoryAgentRegistry()
    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
        llm_gateway=gateway,
    )

    response = await runtime.run(
        "context-agent",
        AgentRequest(
            input="Hello.",
            user_id="user-789",
            session_id="session-789",
        ),
    )

    assert response.session_id == "session-789"
    assert gateway.requests[0]["user_id"] == "user-789"


def assert_gateway_protocol(
    gateway: FakeGateway,
) -> None:
    """
    Runtime-only structural check for the agent-facing Gateway contract.
    """

    gateway_protocol: LLMGateway = gateway
    assert gateway_protocol is gateway


@pytest.mark.asyncio
async def test_runtime_passes_agent_generation_configuration_to_gateway() -> None:
    gateway = FakeGateway()

    definition = AgentDefinition(
        name="configured-agent",
        description="Configured LLM agent",
        system_prompt="You are a configured agent.",
        model="custom-model",
        temperature=0.15,
        max_tokens=768,
    )

    agent = LLMUsingAgent(definition)

    registry = InMemoryAgentRegistry()
    await registry.register(agent)

    runtime = AgentRuntime(
        registry,
        llm_gateway=gateway,
    )

    await runtime.run(
        "configured-agent",
        AgentRequest(input="Analyze this request."),
    )

    assert gateway.requests == [
        {
            "prompt": "Analyze this request.",
            "model": "custom-model",
            "temperature": 0.15,
            "max_tokens": 768,
            "stream": False,
        }
    ]
