from __future__ import annotations

from typing import Any

import pytest

from tests.tools.execution.test_service import FakeTool
from ai_platform.agents.contracts import Agent
from ai_platform.agents.exceptions import AgentToolLoopLimitError
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.llm_context import AgentLLMContext
from ai_platform.agents.llm_config import AgentLLMConfig
from ai_platform.agents.llm_messages import (
    AgentMessageRole,
    assistant_message,
    assistant_tool_call_message,
    system_message,
    tool_message,
    tool_result_message,
    user_message,
)
from ai_platform.agents.models import (
    AgentDefinition,
    AgentRequest,
    AgentResponse,
)
from ai_platform.agents.tool_context import AgentToolContext
from tools.registry.in_memory import InMemoryToolRegistry
from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.tool_calls import AgentToolCall
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)


class FakeLLMGateway:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "Generated answer.",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


class FakeLLMAgent:
    """
    Minimal executable Agent implementation used to validate
    the AgentExecutionContext + AgentLLMContext contract.

    This is intentionally a test fixture, not production code.
    """

    def __init__(self) -> None:
        self._definition = AgentDefinition(
            name="test-llm-agent",
            description="Test LLM-backed agent.",
            system_prompt="You are a test LLM agent.",
            model="mock-gpt",
        )

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        messages = context.build_llm_messages()

        assert messages[0].role is AgentMessageRole.SYSTEM
        assert messages[0].content == self.definition.system_prompt

        result = await context.llm.generate(
            prompt=context.request.input,
            messages=messages,
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


def make_context(
    *,
    request: AgentRequest | None = None,
    history: tuple[Any, ...] = (),
) -> tuple[AgentExecutionContext, FakeLLMGateway]:
    definition = AgentDefinition(
        name="test-llm-agent",
        description="Test LLM-backed agent.",
        system_prompt="You are a test LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        (
            request
            if request is not None
            else AgentRequest(
                input="Hello.",
            )
        ),
        tools=tools,
        llm=llm_context,
        history=history,
    )

    return context, gateway


@pytest.mark.asyncio
async def test_executable_llm_agent_implements_agent_contract() -> None:
    agent: Agent = FakeLLMAgent()

    assert agent.definition.name == "test-llm-agent"
    assert agent.definition.model == "mock-gpt"


@pytest.mark.asyncio
async def test_executable_llm_agent_calls_gateway() -> None:
    agent = FakeLLMAgent()

    context, gateway = make_context(
        request=AgentRequest(
            input="Explain RAG.",
            user_id="user-123",
        ),
    )

    response = await agent.run(context)

    assert response.agent_name == "test-llm-agent"
    assert response.output == "Generated answer."
    assert response.metadata == {
        "provider": "fake",
        "model": "mock-gpt",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    assert gateway.requests == [
        {
            "prompt": "Explain RAG.",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a test LLM agent.",
                },
                {
                    "role": "user",
                    "content": "Explain RAG.",
                },
            ],
            "model": "mock-gpt",
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
            "user_id": "user-123",
        }
    ]


@pytest.mark.asyncio
async def test_executable_llm_agent_preserves_session_id() -> None:
    agent = FakeLLMAgent()

    context, _ = make_context(
        request=AgentRequest(
            input="Continue the conversation.",
            session_id="session-456",
        ),
    )

    response = await agent.run(context)

    assert response.session_id == "session-456"


@pytest.mark.asyncio
async def test_executable_llm_agent_builds_canonical_messages() -> None:
    FakeLLMAgent()

    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    context, _ = make_context(
        request=AgentRequest(
            input="Why is retrieval useful?",
        ),
        history=history,
    )

    messages = context.build_llm_messages()

    assert messages == (
        system_message("You are a test LLM agent."),
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
        user_message("Why is retrieval useful?"),
    )


@pytest.mark.asyncio
async def test_executable_llm_agent_sends_history_to_gateway() -> None:
    agent = FakeLLMAgent()

    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    context, gateway = make_context(
        request=AgentRequest(
            input="Why is retrieval useful?",
        ),
        history=history,
    )

    await agent.run(context)

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a test LLM agent.",
        },
        {
            "role": "user",
            "content": "What is RAG?",
        },
        {
            "role": "assistant",
            "content": "RAG retrieves relevant context.",
        },
        {
            "role": "user",
            "content": "Why is retrieval useful?",
        },
    ]


@pytest.mark.asyncio
async def test_executable_llm_agent_can_build_messages_with_tool_results() -> None:
    context, _ = make_context(
        request=AgentRequest(
            input="What is the pipeline status?",
        ),
    )

    messages = context.build_llm_messages(
        tool_results=('{"status": "healthy"}',),
    )

    assert messages == (
        system_message("You are a test LLM agent."),
        user_message("What is the pipeline status?"),
        tool_message('{"status": "healthy"}'),
    )


@pytest.mark.asyncio
async def test_llm_agent_implements_agent_contract() -> None:
    from ai_platform.agents.contracts import Agent

    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    agent: Agent = LLMAgent(definition)

    assert agent.definition is definition


@pytest.mark.asyncio
async def test_llm_agent_returns_agent_response() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Explain RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    response = await agent.run(context)
    assert response.metadata["tool_rounds"] == 0
    assert response == AgentResponse(
        agent_name="production-llm-agent",
        output="Generated answer.",
        session_id="session-456",
        metadata={
            "provider": "fake",
            "model": "mock-gpt",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "tool_rounds": 0,
        },
    )


@pytest.mark.asyncio
async def test_llm_agent_sends_canonical_messages() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(input="Explain RAG."),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    await agent.run(context)

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Explain RAG.",
        },
    ]


class FakeToolCallingLLMGateway:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        if len(self.requests) == 1:
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="call-123",
                        name="search",
                        arguments={
                            "query": "RAG",
                        },
                    ),
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "RAG retrieves relevant context for generation.",
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "total_tokens": 35,
            },
        }


@pytest.mark.asyncio
async def test_llm_agent_accumulates_tool_call_and_tool_result_messages() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    tool_call = AgentToolCall(
        call_id="call-123",
        name="search",
        arguments={
            "query": "RAG",
        },
    )

    messages = list(context.build_llm_messages())

    await agent._accumulate_tool_call_messages(
        messages,
        context,
        (tool_call,),
        tool_round=1,
        assistant_content="I will search for that.",
    )

    assert messages == [
        system_message("You are a production LLM agent."),
        user_message("Find information about RAG."),
        assistant_tool_call_message(
            tool_calls=(tool_call,),
            content="I will search for that.",
        ),
        tool_result_message(
            call_id="call-123",
            tool_name="search",
            error="Tool not found: search",
        ),
    ]


@pytest.mark.asyncio
async def test_llm_agent_reinvokes_llm_after_tool_execution() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    response = await agent.run(context)

    assert response == AgentResponse(
        agent_name="production-llm-agent",
        output="RAG retrieves relevant context for generation.",
        session_id="session-456",
        metadata={
            "provider": "fake",
            "model": "mock-gpt",
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 10,
                "total_tokens": 35,
            },
            "tool_rounds": 1,
        },
    )

    assert len(gateway.requests) == 2
    assert response.metadata["tool_rounds"] == 1

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
    ]

    assert gateway.requests[1]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
        {
            "role": "assistant",
            "content": "Tool call requested.",
            "tool_calls": [
                {
                    "call_id": "call-123",
                    "name": "search",
                    "arguments": {
                        "query": "RAG",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"call_id": "call-123", "error": "Tool not found: search", "success": false, "tool_name": "search"}',
            "tool_call_id": "call-123",
            "tool_name": "search",
        },
    ]


@pytest.mark.asyncio
async def test_llm_agent_stops_after_max_tool_rounds() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeMultiRoundToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentToolLoopLimitError,
        match="Agent 'production-llm-agent' exceeded the maximum tool-call rounds \\(3\\)",
    ):
        await agent.run(context)

    assert len(gateway.requests) == 4

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
    ]

    assert gateway.requests[1]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
        {
            "role": "assistant",
            "content": "Tool call requested.",
            "tool_calls": [
                {
                    "call_id": "call-1",
                    "name": "search",
                    "arguments": {
                        "query": "round-1",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"call_id": "call-1", "error": "Tool not found: search", "success": false, "tool_name": "search"}',
            "tool_call_id": "call-1",
            "tool_name": "search",
        },
    ]

    assert gateway.requests[2]["messages"] == [
        {
            "role": "system",
            "content": "You are a production LLM agent.",
        },
        {
            "role": "user",
            "content": "Find information about RAG.",
        },
        {
            "role": "assistant",
            "content": "Tool call requested.",
            "tool_calls": [
                {
                    "call_id": "call-1",
                    "name": "search",
                    "arguments": {
                        "query": "round-1",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"call_id": "call-1", "error": "Tool not found: search", "success": false, "tool_name": "search"}',
            "tool_call_id": "call-1",
            "tool_name": "search",
        },
        {
            "role": "assistant",
            "content": "Tool call requested.",
            "tool_calls": [
                {
                    "call_id": "call-2",
                    "name": "search",
                    "arguments": {
                        "query": "round-2",
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": '{"call_id": "call-2", "error": "Tool not found: search", "success": false, "tool_name": "search"}',
            "tool_call_id": "call-2",
            "tool_name": "search",
        },
    ]


class FakeMultiRoundToolCallingLLMGateway:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        round_number = len(self.requests)

        if round_number <= 4:
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id=f"call-{round_number}",
                        name="search",
                        arguments={
                            "query": f"round-{round_number}",
                        },
                    ),
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "This response should never be reached.",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        }


class FakeAgentExecutionObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_llm_agent_emits_event_to_observer() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_STARTED,
        agent_name=definition.name,
    )

    await agent._emit(event)

    assert observer.events == [event]


@pytest.mark.asyncio
async def test_llm_agent_emit_is_noop_without_observer() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    agent = LLMAgent(definition)

    event = AgentExecutionEvent(
        event_type=AgentExecutionEventType.AGENT_STARTED,
        agent_name=definition.name,
    )

    await agent._emit(event)


@pytest.mark.asyncio
async def test_llm_agent_emits_normal_execution_lifecycle_events() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Explain RAG.",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    response = await agent.run(context)

    assert response.output == "Generated answer."

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_STARTED,
        AgentExecutionEventType.LLM_REQUESTED,
        AgentExecutionEventType.LLM_COMPLETED,
        AgentExecutionEventType.AGENT_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_llm_agent_lifecycle_events_include_execution_metadata() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    gateway = FakeLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Explain RAG.",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    await agent.run(context)

    started = observer.events[0]
    llm_requested = observer.events[1]
    llm_completed = observer.events[2]
    completed = observer.events[3]

    assert started.agent_name == "production-llm-agent"
    assert started.session_id == "session-456"

    assert llm_requested.tool_round == 0

    assert llm_completed.tool_round == 0
    assert llm_completed.provider == "fake"
    assert llm_completed.model == "gpt-test"
    assert llm_completed.metadata == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    assert completed.tool_round == 0
    assert completed.provider == "fake"
    assert completed.model == "gpt-test"


@pytest.mark.asyncio
async def test_llm_agent_emits_tool_call_lifecycle_events() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    gateway = FakeToolCallingLLMGateway()

    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(
        FakeTool(name="search"),
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find something.",
            session_id="session-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    await agent.run(context)

    tool_events = [
        event
        for event in observer.events
        if event.event_type
        in {
            AgentExecutionEventType.TOOL_CALL_REQUESTED,
            AgentExecutionEventType.TOOL_CALL_COMPLETED,
            AgentExecutionEventType.TOOL_CALL_FAILED,
        }
    ]

    assert [event.event_type for event in tool_events] == [
        AgentExecutionEventType.TOOL_CALL_REQUESTED,
        AgentExecutionEventType.TOOL_CALL_COMPLETED,
    ]

    assert tool_events[0].agent_name == definition.name
    assert tool_events[0].session_id == "session-123"
    assert tool_events[0].tool_round == 1
    assert tool_events[0].tool_name == "search"
    assert tool_events[0].call_id == "call-123"


@pytest.mark.asyncio
async def test_llm_agent_tool_events_do_not_capture_tool_payloads() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    # Reuse the same gateway/tool setup as the previous test.
    gateway = FakeToolCallingLLMGateway()
    tool_registry = InMemoryToolRegistry()

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find something.",
            session_id="session-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    await agent.run(context)

    tool_events = [
        event
        for event in observer.events
        if event.event_type
        in {
            AgentExecutionEventType.TOOL_CALL_REQUESTED,
            AgentExecutionEventType.TOOL_CALL_COMPLETED,
            AgentExecutionEventType.TOOL_CALL_FAILED,
        }
    ]

    for event in tool_events:
        assert event.metadata == {}


@pytest.mark.asyncio
async def test_llm_agent_emits_complete_lifecycle_after_tool_execution() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    gateway = FakeToolCallingLLMGateway()
    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(
        FakeTool(name="search"),
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find something.",
            session_id="session-123",
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    response = await agent.run(context)

    assert response.output == "RAG retrieves relevant context for generation."

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_STARTED,
        AgentExecutionEventType.LLM_REQUESTED,
        AgentExecutionEventType.LLM_COMPLETED,
        AgentExecutionEventType.TOOL_CALL_REQUESTED,
        AgentExecutionEventType.TOOL_CALL_COMPLETED,
        AgentExecutionEventType.LLM_REQUESTED,
        AgentExecutionEventType.LLM_COMPLETED,
        AgentExecutionEventType.AGENT_COMPLETED,
    ]

    first_llm_completed = observer.events[2]
    final_llm_completed = observer.events[6]
    agent_completed = observer.events[7]

    assert first_llm_completed.tool_round == 0
    assert first_llm_completed.provider == "fake"
    assert first_llm_completed.model == "gpt-test"
    assert first_llm_completed.metadata == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    assert final_llm_completed.tool_round == 1
    assert final_llm_completed.provider == "fake"
    assert final_llm_completed.model == "gpt-test"
    assert final_llm_completed.metadata == {
        "prompt_tokens": 25,
        "completion_tokens": 10,
        "total_tokens": 35,
    }

    assert agent_completed.tool_round == 1
    assert agent_completed.provider == "fake"
    assert agent_completed.model == "gpt-test"


@pytest.mark.asyncio
async def test_llm_agent_emits_agent_failed_on_llm_failure() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
    )

    class FailingLLMGateway:
        async def route_chat(
            self,
            request: dict[str, Any],
        ) -> dict[str, Any]:
            raise RuntimeError("LLM provider failure")

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Explain RAG.",
            session_id="session-failure",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            FailingLLMGateway(),
            definition.llm_config,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="LLM provider failure",
    ):
        await agent.run(context)

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_STARTED,
        AgentExecutionEventType.LLM_REQUESTED,
        AgentExecutionEventType.AGENT_FAILED,
    ]

    failed = observer.events[-1]

    assert failed.agent_name == definition.name
    assert failed.session_id == "session-failure"
    assert failed.tool_round == 0
    assert failed.metadata == {
        "error_type": "RuntimeError",
    }


@pytest.mark.asyncio
async def test_llm_agent_emits_agent_failed_on_tool_loop_limit() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    gateway = FakeMultiRoundToolCallingLLMGateway()
    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-loop-limit",
        ),
        tools=AgentToolContext(
            InMemoryToolRegistry(),
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
    )

    with pytest.raises(
        AgentToolLoopLimitError,
        match="Agent 'production-llm-agent' exceeded the maximum tool-call rounds \\(3\\)",
    ):
        await agent.run(context)

    assert observer.events[-1].event_type == AgentExecutionEventType.AGENT_FAILED
    assert observer.events[-1].agent_name == definition.name
    assert observer.events[-1].session_id == "session-loop-limit"
    assert observer.events[-1].tool_round == 3
    assert observer.events[-1].metadata == {
        "error_type": "AgentToolLoopLimitError",
    }
