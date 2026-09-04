from __future__ import annotations

import pytest

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.llm_context import AgentLLMContext
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.tool_context import AgentToolContext
from ai_platform.agents.tool_calls import (
    AgentToolCall,
    AgentToolResult,
)
from tools.registry.in_memory import InMemoryToolRegistry
from ai_platform.agents.llm_messages import (
    assistant_message,
    system_message,
    tool_message,
    user_message,
)


class FakeGateway:
    async def route_chat(self, request: dict):
        return {
            "provider": "mock",
            "reply": "test",
            "model": request["model"],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }


def make_definition(
    *,
    tool_names: tuple[str, ...] = (),
) -> AgentDefinition:
    return AgentDefinition(
        name="test-agent",
        description="Test agent",
        system_prompt="You are a test agent.",
        tool_names=tool_names,
    )


def make_context(
    request: AgentRequest | None = None,
    *,
    tool_names: tuple[str, ...] = (),
) -> AgentExecutionContext:
    definition = make_definition(
        tool_names=tool_names,
    )

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
    )

    llm = AgentLLMContext(
        FakeGateway(),
        definition.llm_config,
    )

    return AgentExecutionContext(
        request or AgentRequest(input="Hello."),
        tools=tools,
        llm=llm,
    )


@pytest.mark.asyncio
async def test_execution_context_preserves_request() -> None:
    request = AgentRequest(
        input="Hello.",
        session_id="session-123",
        user_id="user-456",
    )

    context = make_context(request)

    assert context.request is request
    assert context.session_id == "session-123"
    assert context.user_id == "user-456"


@pytest.mark.asyncio
async def test_execution_context_exposes_tool_context() -> None:
    context = make_context(
        tool_names=("search",),
    )

    assert context.tools.agent_name == "test-agent"
    assert context.tools.tool_names == ("search",)


@pytest.mark.asyncio
async def test_execution_context_exposes_llm_context() -> None:
    context = make_context()

    assert context.llm is not None


@pytest.mark.asyncio
async def test_execution_context_llm_context_can_generate() -> None:
    context = make_context()

    response = await context.llm.generate(
        prompt="Explain RAG.",
        model="mock-gpt",
    )

    assert response.text == "test"
    assert response.provider == "mock"
    assert response.model == "mock-gpt"
    assert response.usage.total_tokens == 2


@pytest.mark.asyncio
async def test_execution_context_keeps_request_and_capabilities_separate() -> None:
    request = AgentRequest(
        input="Use the available capabilities.",
        session_id="session-789",
    )

    context = make_context(
        request,
        tool_names=("search",),
    )

    assert context.request is request
    assert context.tools is not None
    assert context.llm is not None
    assert context.request is not context.tools
    assert context.request is not context.llm


@pytest.mark.asyncio
async def test_execution_context_defaults_to_empty_history() -> None:
    context = make_context()

    assert context.history == ()


@pytest.mark.asyncio
async def test_execution_context_preserves_history() -> None:
    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    context = make_context()

    context = AgentExecutionContext(
        context.request,
        tools=context.tools,
        llm=context.llm,
        history=history,
    )

    assert context.history == history
    assert context.history[0].content == "What is RAG?"
    assert context.history[1].content == ("RAG retrieves relevant context.")


@pytest.mark.asyncio
async def test_execution_context_rejects_invalid_history() -> None:
    base_context = make_context()

    with pytest.raises(
        TypeError,
        match="Agent execution history must contain AgentMessage instances",
    ):
        AgentExecutionContext(
            base_context.request,
            tools=base_context.tools,
            llm=base_context.llm,
            history=("invalid",),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_execution_context_builds_llm_messages() -> None:
    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Why is retrieval useful?",
        ),
        tools=make_context().tools,
        llm=make_context().llm,
        history=history,
    )

    messages = context.build_llm_messages()

    assert messages == (
        system_message("You are a test agent."),
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
        user_message("Why is retrieval useful?"),
    )


@pytest.mark.asyncio
async def test_execution_context_builds_llm_messages_with_tool_results() -> None:
    context = make_context(
        AgentRequest(
            input="What is the pipeline status?",
        ),
    )

    messages = context.build_llm_messages(
        tool_results=('{"status": "healthy"}',),
    )

    assert messages == (
        system_message("You are a test agent."),
        user_message("What is the pipeline status?"),
        tool_message('{"status": "healthy"}'),
    )


@pytest.mark.asyncio
async def test_execution_context_builds_llm_messages_without_history() -> None:
    context = make_context(
        AgentRequest(
            input="Hello.",
        ),
    )

    messages = context.build_llm_messages()

    assert messages == (
        system_message("You are a test agent."),
        user_message("Hello."),
    )


@pytest.mark.asyncio
async def test_execution_context_builds_tool_result_messages() -> None:
    context = make_context()

    results = (
        AgentToolResult(
            call_id="call-123",
            tool_name="search",
            output={
                "results": [
                    "document-1",
                    "document-2",
                ]
            },
        ),
    )

    messages = await context.build_tool_result_messages(results)

    assert len(messages) == 1
    assert messages[0].role.value == "tool"
    assert messages[0].content == (
        '{"call_id": "call-123", "output": '
        '{"results": ["document-1", "document-2"]}, '
        '"success": true, "tool_name": "search"}'
    )


@pytest.mark.asyncio
async def test_execution_context_builds_failed_tool_result_messages() -> None:
    context = make_context()

    results = (
        AgentToolResult(
            call_id="call-456",
            tool_name="search",
            error="Search service unavailable.",
        ),
    )

    messages = await context.build_tool_result_messages(results)

    assert len(messages) == 1
    assert messages[0].role.value == "tool"
    assert messages[0].content == (
        '{"call_id": "call-456", '
        '"error": "Search service unavailable.", '
        '"success": false, "tool_name": "search"}'
    )


@pytest.mark.asyncio
async def test_execution_context_preserves_tool_result_order() -> None:
    context = make_context()

    results = (
        AgentToolResult(
            call_id="call-1",
            tool_name="search",
            output="first",
        ),
        AgentToolResult(
            call_id="call-2",
            tool_name="status",
            output="second",
        ),
    )

    messages = await context.build_tool_result_messages(results)

    assert len(messages) == 2

    assert '"call_id": "call-1"' in messages[0].content
    assert '"tool_name": "search"' in messages[0].content

    assert '"call_id": "call-2"' in messages[1].content
    assert '"tool_name": "status"' in messages[1].content


@pytest.mark.asyncio
async def test_execution_context_rejects_invalid_tool_results() -> None:
    context = make_context()

    with pytest.raises(
        TypeError,
        match="Tool results must contain AgentToolResult instances",
    ):
        await context.build_tool_result_messages(
            ("invalid",),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_execution_context_executes_tool_calls() -> None:
    context = make_context(
        tool_names=("search",),
    )

    context.tools._execution_service

    result = await context.execute_tool_calls(
        (
            AgentToolCall(
                call_id="call-123",
                name="search",
                arguments={"query": "RAG"},
            ),
        )
    )

    assert result == (
        AgentToolResult(
            call_id="call-123",
            tool_name="search",
            error="Tool not found: search",
        ),
    )


class FakeToolExecutionService:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def execute(
        self,
        tool_name: str,
        arguments: dict,
        *,
        principal: str | None = None,
        timeout_seconds: float | None = None,
    ) -> dict:
        self.calls.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "principal": principal,
                "timeout_seconds": timeout_seconds,
            }
        )

        return {
            "success": True,
            "output": {
                "status": "healthy",
            },
        }


@pytest.mark.asyncio
async def test_execution_context_maps_tool_call_to_tool_result() -> None:
    definition = make_definition(
        tool_names=("search",),
    )

    execution_service = FakeToolExecutionService()

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
        execution_service=execution_service,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Check the status.",
            user_id="user-123",
        ),
        tools=tools,
        llm=AgentLLMContext(
            FakeGateway(),
            definition.llm_config,
        ),
    )

    results = await context.execute_tool_calls(
        (
            AgentToolCall(
                call_id="call-123",
                name="search",
                arguments={
                    "query": "pipeline status",
                },
            ),
        )
    )

    assert results == (
        AgentToolResult(
            call_id="call-123",
            tool_name="search",
            output={
                "status": "healthy",
            },
        ),
    )

    assert execution_service.calls == [
        {
            "tool_name": "search",
            "arguments": {
                "query": "pipeline status",
            },
            "principal": "user-123",
            "timeout_seconds": None,
        }
    ]


@pytest.mark.asyncio
async def test_execution_context_preserves_tool_call_order() -> None:
    definition = make_definition(
        tool_names=("search", "status"),
    )

    execution_service = FakeToolExecutionService()

    tools = AgentToolContext(
        InMemoryToolRegistry(),
        definition,
        execution_service=execution_service,
    )

    context = AgentExecutionContext(
        AgentRequest(input="Check everything."),
        tools=tools,
        llm=AgentLLMContext(
            FakeGateway(),
            definition.llm_config,
        ),
    )

    results = await context.execute_tool_calls(
        (
            AgentToolCall(
                call_id="call-1",
                name="search",
                arguments={"query": "RAG"},
            ),
            AgentToolCall(
                call_id="call-2",
                name="status",
                arguments={},
            ),
        )
    )

    assert [result.call_id for result in results] == [
        "call-1",
        "call-2",
    ]

    assert [result.tool_name for result in results] == [
        "search",
        "status",
    ]


@pytest.mark.asyncio
async def test_execution_context_rejects_invalid_tool_calls() -> None:
    context = make_context()

    with pytest.raises(
        TypeError,
        match="Tool calls must contain AgentToolCall instances",
    ):
        await context.execute_tool_calls(
            ("invalid",),  # type: ignore[arg-type]
        )
