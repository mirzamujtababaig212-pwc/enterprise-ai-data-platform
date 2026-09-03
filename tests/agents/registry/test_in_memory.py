from __future__ import annotations

import pytest

from ai_platform.agents.models import (
    AgentDefinition,
    AgentRequest,
    AgentResponse,
)
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry


class FakeAgent:
    def __init__(
        self,
        definition: AgentDefinition,
        output: str = "test output",
    ) -> None:
        self._definition = definition
        self._output = output

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        request: AgentRequest,
    ) -> AgentResponse:
        return AgentResponse(
            agent_name=self.definition.name,
            output=self._output,
            session_id=request.session_id,
        )


def make_agent(
    name: str = "test-agent",
    *,
    enabled: bool = True,
    description: str = "Test agent.",
    output: str = "test output",
) -> FakeAgent:
    return FakeAgent(
        AgentDefinition(
            name=name,
            description=description,
            system_prompt="You are a test agent.",
            model="test-model",
            tool_names=("search_documents",),
            enabled=enabled,
        ),
        output=output,
    )


@pytest.mark.asyncio
async def test_register_and_get_agent():
    registry = InMemoryAgentRegistry()

    agent = make_agent(
        name="data-engineering-agent",
    )

    await registry.register(agent)

    result = await registry.get(
        "data-engineering-agent",
    )

    assert result is agent


@pytest.mark.asyncio
async def test_get_unknown_agent_returns_none():
    registry = InMemoryAgentRegistry()

    result = await registry.get(
        "unknown-agent",
    )

    assert result is None


@pytest.mark.asyncio
async def test_register_multiple_agents():
    registry = InMemoryAgentRegistry()

    first = make_agent(
        name="data-engineering-agent",
    )
    second = make_agent(
        name="devops-agent",
    )

    await registry.register(first)
    await registry.register(second)

    assert (
        await registry.get(
            "data-engineering-agent",
        )
        is first
    )

    assert (
        await registry.get(
            "devops-agent",
        )
        is second
    )


@pytest.mark.asyncio
async def test_register_same_name_replaces_existing_agent():
    registry = InMemoryAgentRegistry()

    first = make_agent(
        name="test-agent",
    )
    second = make_agent(
        name="test-agent",
    )

    await registry.register(first)
    await registry.register(second)

    result = await registry.get(
        "test-agent",
    )

    assert result is second


@pytest.mark.asyncio
async def test_list_agents_returns_enabled_agents():
    registry = InMemoryAgentRegistry()

    enabled_agent = make_agent(
        name="enabled-agent",
        enabled=True,
    )

    disabled_agent = make_agent(
        name="disabled-agent",
        enabled=False,
    )

    await registry.register(enabled_agent)
    await registry.register(disabled_agent)

    definitions = await registry.list_agents()

    assert definitions == [
        enabled_agent.definition,
    ]


@pytest.mark.asyncio
async def test_list_agents_preserves_registration_order():
    registry = InMemoryAgentRegistry()

    first = make_agent(
        name="first-agent",
    )
    second = make_agent(
        name="second-agent",
    )
    third = make_agent(
        name="third-agent",
    )

    await registry.register(first)
    await registry.register(second)
    await registry.register(third)

    definitions = await registry.list_agents()

    assert [definition.name for definition in definitions] == [
        "first-agent",
        "second-agent",
        "third-agent",
    ]


@pytest.mark.asyncio
async def test_remove_agent():
    registry = InMemoryAgentRegistry()

    agent = make_agent(
        name="test-agent",
    )

    await registry.register(agent)

    assert (
        await registry.get(
            "test-agent",
        )
        is agent
    )

    await registry.remove(
        "test-agent",
    )

    assert (
        await registry.get(
            "test-agent",
        )
        is None
    )


@pytest.mark.asyncio
async def test_remove_unknown_agent_is_noop():
    registry = InMemoryAgentRegistry()

    await registry.remove(
        "unknown-agent",
    )

    assert await registry.list_agents() == []


@pytest.mark.asyncio
async def test_get_rejects_empty_name():
    registry = InMemoryAgentRegistry()

    with pytest.raises(
        ValueError,
        match="Agent name must not be empty",
    ):
        await registry.get("")


@pytest.mark.asyncio
async def test_get_rejects_whitespace_name():
    registry = InMemoryAgentRegistry()

    with pytest.raises(
        ValueError,
        match="Agent name must not be empty",
    ):
        await registry.get("   ")


@pytest.mark.asyncio
async def test_remove_rejects_empty_name():
    registry = InMemoryAgentRegistry()

    with pytest.raises(
        ValueError,
        match="Agent name must not be empty",
    ):
        await registry.remove("")


@pytest.mark.asyncio
async def test_remove_rejects_whitespace_name():
    registry = InMemoryAgentRegistry()

    with pytest.raises(
        ValueError,
        match="Agent name must not be empty",
    ):
        await registry.remove("   ")


@pytest.mark.asyncio
async def test_register_does_not_execute_agent():
    registry = InMemoryAgentRegistry()

    agent = make_agent(
        name="test-agent",
    )

    await registry.register(agent)

    result = await registry.get(
        "test-agent",
    )

    assert result is agent


@pytest.mark.asyncio
async def test_list_agents_returns_definitions_not_agents():
    registry = InMemoryAgentRegistry()

    agent = make_agent(
        name="test-agent",
    )

    await registry.register(agent)

    definitions = await registry.list_agents()

    assert len(definitions) == 1
    assert definitions[0] is agent.definition
    assert isinstance(
        definitions[0],
        AgentDefinition,
    )


@pytest.mark.asyncio
async def test_disabled_agent_can_still_be_retrieved():
    registry = InMemoryAgentRegistry()

    agent = make_agent(
        name="disabled-agent",
        enabled=False,
    )

    await registry.register(agent)

    result = await registry.get(
        "disabled-agent",
    )

    assert result is agent

    assert await registry.list_agents() == []


@pytest.mark.asyncio
async def test_registry_starts_empty():
    registry = InMemoryAgentRegistry()

    assert await registry.list_agents() == []
    assert await registry.get("test-agent") is None


@pytest.mark.asyncio
async def test_agent_can_execute_after_registry_lookup():
    registry = InMemoryAgentRegistry()

    agent = make_agent(
        name="test-agent",
        output="Hello from registry agent.",
    )

    await registry.register(agent)

    registered = await registry.get(
        "test-agent",
    )

    assert registered is not None

    response = await registered.run(
        AgentRequest(
            input="Hello.",
            session_id="session-123",
        )
    )

    assert response.agent_name == "test-agent"
    assert response.output == "Hello from registry agent."
    assert response.session_id == "session-123"
