from __future__ import annotations

import asyncio

from typing import Any

import pytest

from tests.tools.execution.test_service import FakeTool, FailingTool
from ai_platform.agents.contracts import Agent
from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.budget import ExecutionBudget, ExecutionBudgetState
from ai_platform.agents.exceptions import (
    AgentExecutionOwnershipLostError,
    AgentLLMCallLimitError,
    AgentToolLoopLimitError,
)
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
from tools.models import ToolDefinition
from tools.rag.search import RAGSearchTool
from tools.execution.idempotency import InMemoryToolExecutionIdempotencyStore
from tools.execution.service import ToolExecutionService
from tools.execution.context import ToolExecutionContext
from rag.governance import GovernancePolicy
from rag.models import DocumentChunk, RetrievalResult
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


class FakeRAGTool:
    def __init__(self, name: str = "rag.search") -> None:
        self._definition = ToolDefinition(
            name=name,
            description="A test RAG search tool.",
        )
        self.execute_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execute_count += 1

        return {
            "query": arguments["query"],
            "retrieved_count": 2,
            "results": [
                {
                    "chunk_id": "chunk-rag-001",
                    "document_id": "doc-rag-001",
                    "content": "Sensitive enterprise context.",
                    "score": 0.91,
                    "metadata": {
                        "classification": "confidential",
                    },
                },
                {
                    "chunk_id": "chunk-rag-002",
                    "document_id": "doc-rag-002",
                    "content": "Another sensitive context.",
                    "score": 0.83,
                    "metadata": {
                        "classification": "restricted",
                    },
                },
            ],
        }


class FakeToolCallingLLMGateway:
    def __init__(self, tool_name: str = "search") -> None:
        self.requests: list[dict[str, Any]] = []
        self.tool_name = tool_name

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
                        name=self.tool_name,
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


class OwnershipLossDuringLLMGateway(FakeToolCallingLLMGateway):
    def __init__(self, ownership_lost: asyncio.Event) -> None:
        super().__init__()
        self.ownership_lost = ownership_lost

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        response = await super().route_chat(request)
        self.ownership_lost.set()
        return response


@pytest.mark.asyncio
async def test_llm_agent_stops_before_llm_when_execution_ownership_is_already_lost() -> None:
    ownership_lost = asyncio.Event()
    ownership_lost.set()

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
            input="Find information about RAG.",
            user_id="user-123",
        ),
        tools=tools,
        llm=llm_context,
        execution_ownership_lost=ownership_lost,
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentExecutionOwnershipLostError,
        match="lost durable run ownership",
    ):
        await agent.run(context)

    assert gateway.requests == []


@pytest.mark.asyncio
async def test_llm_agent_does_not_execute_tool_after_ownership_is_lost_during_llm_call() -> None:
    ownership_lost = asyncio.Event()

    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = OwnershipLossDuringLLMGateway(ownership_lost)

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool = FakeTool(name="search")

    registry = InMemoryToolRegistry()
    await registry.register(tool)

    tools = AgentToolContext(
        registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
        ),
        tools=tools,
        llm=llm_context,
        execution_ownership_lost=ownership_lost,
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentExecutionOwnershipLostError,
        match="lost durable run ownership",
    ):
        await agent.run(context)

    assert len(gateway.requests) == 1
    assert tool.execution_count == 0


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


class FakeResumeContinuationLLMGateway:
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
                    "prompt_tokens": 20,
                    "completion_tokens": 5,
                    "total_tokens": 25,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="call-2",
                        name="search",
                        arguments={
                            "query": "follow-up",
                        },
                    ),
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "Resumed execution completed successfully.",
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "total_tokens": 40,
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


class FakeAgentCheckpointHandler:
    def __init__(self) -> None:
        self.checkpoints: list[AgentExecutionCheckpoint] = []

    async def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
        *,
        lease_id: str | None = None,
    ) -> None:
        self.checkpoints.append(checkpoint)


class FailingAgentCheckpointHandler:
    def __init__(
        self,
        *,
        fail_position: AgentCheckpointPosition = AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
    ) -> None:
        self.fail_position = fail_position
        self.attempted_checkpoints: list[AgentExecutionCheckpoint] = []

    async def save(
        self,
        checkpoint: AgentExecutionCheckpoint,
        *,
        lease_id: str | None = None,
    ) -> None:
        self.attempted_checkpoints.append(checkpoint)
        if checkpoint.position is self.fail_position:
            raise RuntimeError("checkpoint persistence failed")


@pytest.mark.asyncio
async def test_llm_agent_captures_checkpoint_after_tool_execution() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FakeAgentCheckpointHandler()
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
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-123",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    await agent.run(context)

    assert len(checkpoint_handler.checkpoints) == 2

    before_checkpoint = checkpoint_handler.checkpoints[0]

    assert before_checkpoint.run_id == "run-123"
    assert before_checkpoint.agent_name == "production-llm-agent"
    assert before_checkpoint.session_id == "session-456"
    assert before_checkpoint.user_id == "user-123"
    assert before_checkpoint.tool_round == 1
    assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert before_checkpoint.metadata == {"request_id": "request-789"}
    assert before_checkpoint.messages[-1].role is AgentMessageRole.ASSISTANT
    assert "call-123" in before_checkpoint.messages[-1].content
    assert len(before_checkpoint.messages) == 3

    after_checkpoint = checkpoint_handler.checkpoints[1]

    assert after_checkpoint.run_id == "run-123"
    assert after_checkpoint.agent_name == "production-llm-agent"
    assert after_checkpoint.session_id == "session-456"
    assert after_checkpoint.user_id == "user-123"
    assert after_checkpoint.tool_round == 1
    assert after_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert after_checkpoint.metadata == {"request_id": "request-789"}

    assert after_checkpoint.messages[-1].role is AgentMessageRole.TOOL
    assert "call-123" in after_checkpoint.messages[-1].content
    assert len(after_checkpoint.messages) == 4


@pytest.mark.asyncio
async def test_llm_agent_checkpoint_failure_before_tool_prevents_side_effect() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FailingAgentCheckpointHandler(
        fail_position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
    )
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-checkpoint-before-tool-failure-1",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint persistence failed",
    ):
        await agent.run(context)

    assert tool.execute_count == 0
    assert len(gateway.requests) == 1
    assert len(checkpoint_handler.attempted_checkpoints) == 1

    attempted_checkpoint = checkpoint_handler.attempted_checkpoints[0]

    assert attempted_checkpoint.run_id == "run-checkpoint-before-tool-failure-1"
    assert attempted_checkpoint.tool_round == 1
    assert attempted_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert attempted_checkpoint.messages[-1].role is AgentMessageRole.ASSISTANT
    assert "call-123" in attempted_checkpoint.messages[-1].content


@pytest.mark.asyncio
async def test_llm_agent_checkpoint_failure_occurs_after_tool_side_effect() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FailingAgentCheckpointHandler()
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-checkpoint-failure-1",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint persistence failed",
    ):
        await agent.run(context)

    assert tool.execute_count == 1
    assert len(gateway.requests) == 1

    assert len(checkpoint_handler.attempted_checkpoints) == 2

    before_checkpoint = checkpoint_handler.attempted_checkpoints[0]

    assert before_checkpoint.run_id == "run-checkpoint-failure-1"
    assert before_checkpoint.tool_round == 1
    assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert before_checkpoint.messages[-1].role is AgentMessageRole.ASSISTANT
    assert "call-123" in before_checkpoint.messages[-1].content

    after_checkpoint = checkpoint_handler.attempted_checkpoints[1]

    assert after_checkpoint.run_id == "run-checkpoint-failure-1"
    assert after_checkpoint.tool_round == 1
    assert after_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert after_checkpoint.messages[-1].role is AgentMessageRole.TOOL
    assert "call-123" in after_checkpoint.messages[-1].content


@pytest.mark.asyncio
async def test_llm_agent_checkpoint_failure_preserves_idempotent_tool_result() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FailingAgentCheckpointHandler()
    gateway = FakeToolCallingLLMGateway()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    idempotency_store = InMemoryToolExecutionIdempotencyStore()
    execution_service = ToolExecutionService(
        tool_registry,
        idempotency_store=idempotency_store,
    )

    tools = AgentToolContext(
        tool_registry,
        definition,
        execution_service=execution_service,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-checkpoint-idempotency-1",
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    with pytest.raises(
        RuntimeError,
        match="checkpoint persistence failed",
    ):
        await agent.run(context)

    assert tool.execute_count == 1
    assert len(gateway.requests) == 1
    assert len(checkpoint_handler.attempted_checkpoints) == 2

    before_checkpoint = checkpoint_handler.attempted_checkpoints[0]

    assert before_checkpoint.run_id == "run-checkpoint-idempotency-1"
    assert before_checkpoint.tool_round == 1
    assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert before_checkpoint.messages[-1].role is AgentMessageRole.ASSISTANT
    assert "call-123" in before_checkpoint.messages[-1].content

    after_checkpoint = checkpoint_handler.attempted_checkpoints[1]

    assert after_checkpoint.run_id == "run-checkpoint-idempotency-1"
    assert after_checkpoint.tool_round == 1
    assert after_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert after_checkpoint.messages[-1].role is AgentMessageRole.TOOL
    assert "call-123" in after_checkpoint.messages[-1].content

    replay_service = ToolExecutionService(
        tool_registry,
        idempotency_store=idempotency_store,
    )

    replay_context = ToolExecutionContext(
        run_id="run-checkpoint-idempotency-1",
        call_id="call-123",
        agent_name="production-llm-agent",
        session_id="session-456",
        user_id="user-123",
        request_metadata={"request_id": "request-789"},
    )

    replay = await replay_service.execute(
        "search",
        {"query": "RAG"},
        execution_context=replay_context,
    )

    assert replay.success is True
    assert replay.output == {
        "query": "RAG",
        "retrieved_count": 2,
        "results": [
            {
                "chunk_id": "chunk-rag-001",
                "document_id": "doc-rag-001",
                "content": "Sensitive enterprise context.",
                "score": 0.91,
                "metadata": {
                    "classification": "confidential",
                },
            },
            {
                "chunk_id": "chunk-rag-002",
                "document_id": "doc-rag-002",
                "content": "Another sensitive context.",
                "score": 0.83,
                "metadata": {
                    "classification": "restricted",
                },
            },
        ],
    }

    assert tool.execute_count == 1


@pytest.mark.asyncio
async def test_llm_agent_does_not_capture_checkpoint_without_run_id() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("search",),
    )

    checkpoint_handler = FakeAgentCheckpointHandler()
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

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    await agent.run(context)

    assert checkpoint_handler.checkpoints == []


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
            user_id="user-123",
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
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-123",
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
    assert started.user_id == "user-123"

    assert llm_requested.tool_round == 0
    assert llm_requested.user_id == "user-123"

    assert llm_completed.tool_round == 0
    assert llm_completed.user_id == "user-123"
    assert llm_completed.provider == "fake"
    assert llm_completed.model == "gpt-test"
    assert llm_completed.metadata == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }

    assert completed.tool_round == 0
    assert completed.user_id == "user-123"
    assert completed.provider == "fake"
    assert completed.model == "gpt-test"

    assert [event.run_id for event in observer.events] == [
        "run-123",
        "run-123",
        "run-123",
        "run-123",
    ]


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
async def test_llm_agent_rag_tool_event_captures_sanitized_provenance() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(FakeRAGTool())

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-123",
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

    completed = next(
        event
        for event in observer.events
        if event.event_type == AgentExecutionEventType.TOOL_CALL_COMPLETED
    )

    assert completed.tool_name == "rag.search"
    assert completed.call_id == "call-123"
    assert completed.metadata == {
        "rag_provenance": {
            "retrieved_count": 2,
            "sources": [
                {
                    "chunk_id": "chunk-rag-001",
                    "document_id": "doc-rag-001",
                    "score": 0.91,
                },
                {
                    "chunk_id": "chunk-rag-002",
                    "document_id": "doc-rag-002",
                    "score": 0.83,
                },
            ],
        }
    }

    serialized = str(completed.metadata)

    assert "Sensitive enterprise context." not in serialized
    assert "Another sensitive context." not in serialized
    assert "classification" not in serialized


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
        if event.event_type is AgentExecutionEventType.TOOL_CALL_FAILED:
            assert event.metadata == {
                "failure_category": "tool_not_found",
            }
        else:
            assert event.metadata == {}


@pytest.mark.asyncio
async def test_llm_agent_failed_tool_event_includes_failure_category() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("failing_tool",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="failing_tool")
    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(FailingTool())

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Run the failing tool.",
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

    failed_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.TOOL_CALL_FAILED
    ]

    assert len(failed_events) == 1
    failed_event = failed_events[0]

    assert failed_event.metadata == {
        "failure_category": "execution_error",
    }
    assert "simulated tool failure" not in str(failed_event.metadata)


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
async def test_llm_agent_executes_real_rag_tool_with_governance_context() -> None:
    class FakeRetriever:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def retrieve(
            self,
            query: str,
            top_k: int,
            min_score: float | None = None,
            metadata_filter: dict[str, object] | None = None,
            governance_policy: GovernancePolicy | None = None,
        ) -> list[Any]:
            self.calls.append(
                {
                    "query": query,
                    "top_k": top_k,
                    "min_score": min_score,
                    "metadata_filter": metadata_filter,
                    "governance_policy": governance_policy,
                }
            )

            return [
                RetrievalResult(
                    chunk=DocumentChunk(
                        id="chunk-rag-001",
                        document_id="doc-rag-001",
                        content="Enterprise RAG retrieves relevant knowledge.",
                        metadata={"source": "test"},
                        chunk_index=0,
                    ),
                    score=0.95,
                )
            ]

    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production assistant.",
        model="gpt-test",
        tool_names=("rag.search",),
    )

    gateway = FakeToolCallingLLMGateway(tool_name="rag.search")
    retriever = FakeRetriever()
    tool_registry = InMemoryToolRegistry()

    await tool_registry.register(
        RAGSearchTool(retriever),
    )

    policy = GovernancePolicy(
        required_metadata={
            "tenant_id": "tenant-a",
        }
    )

    observer = FakeAgentExecutionObserver()

    agent = LLMAgent(
        definition,
        observer=observer,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            session_id="session-rag-123",
            user_id="user-rag-456",
            governance_policy=policy,
            metadata={
                "source": "agent-api",
            },
        ),
        tools=AgentToolContext(
            tool_registry,
            definition,
        ),
        llm=AgentLLMContext(
            gateway,
            definition.llm_config,
        ),
        run_id="run-rag-123",
    )

    response = await agent.run(context)

    assert response.output == "RAG retrieves relevant context for generation."

    assert retriever.calls == [
        {
            "query": "RAG",
            "top_k": 5,
            "min_score": None,
            "metadata_filter": None,
            "governance_policy": policy,
        }
    ]

    completed = next(
        event
        for event in observer.events
        if event.event_type == AgentExecutionEventType.TOOL_CALL_COMPLETED
    )

    assert completed.tool_name == "rag.search"
    assert completed.call_id == "call-123"
    assert completed.run_id == "run-rag-123"
    assert completed.session_id == "session-rag-123"


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


@pytest.mark.asyncio
async def test_llm_agent_resume_uses_checkpoint_messages_without_replaying_tool() -> None:
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

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
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
        run_id="run-resume-1",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-resume-1",
        agent_name=definition.name,
        session_id="session-456",
        user_id="user-123",
        messages=(
            system_message("You are a production LLM agent."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-completed-1",
                        name="search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
            tool_result_message(
                call_id="call-completed-1",
                tool_name="search",
                output={
                    "query": "RAG",
                    "retrieved_count": 2,
                },
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    agent = LLMAgent(definition)

    response = await agent.resume(context, checkpoint)

    assert response.output == "Generated answer."
    assert response.metadata["tool_rounds"] == 1
    assert tool.execute_count == 0

    assert len(gateway.requests) == 1
    assert gateway.requests[0]["messages"] == [
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
            "content": "I searched for the information.",
            "tool_calls": [
                {
                    "call_id": "call-completed-1",
                    "name": "search",
                    "arguments": {"query": "RAG"},
                }
            ],
        },
        {
            "role": "tool",
            "content": (
                '{"call_id": "call-completed-1", '
                '"output": {"query": "RAG", "retrieved_count": 2}, '
                '"success": true, "tool_name": "search"}'
            ),
            "tool_call_id": "call-completed-1",
            "tool_name": "search",
        },
    ]


@pytest.mark.asyncio
async def test_llm_agent_resume_from_before_tool_checkpoint_executes_saved_tool() -> None:
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

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    checkpoint_handler = FakeAgentCheckpointHandler()

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-before-tool-resume-1",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-before-tool-resume-1",
        agent_name=definition.name,
        session_id="session-456",
        user_id="user-123",
        messages=(
            system_message("You are a production LLM agent."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-recover-1",
                        name="search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the information.",
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
        metadata={},
        execution_budget_state=ExecutionBudgetState(
            llm_calls=1,
            tool_calls=1,
            tool_rounds=1,
        ),
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    response = await agent.resume(context, checkpoint)

    assert response.output == "Generated answer."
    assert response.metadata["tool_rounds"] == 1

    assert tool.execute_count == 1

    # Recovery must execute the saved tool before making the next LLM call.
    assert len(gateway.requests) == 1
    assert gateway.requests[0]["messages"][-1]["role"] == "tool"
    assert gateway.requests[0]["messages"][-1]["tool_call_id"] == "call-recover-1"
    assert gateway.requests[0]["messages"][-1]["tool_name"] == "search"
    assert '"query": "RAG"' in gateway.requests[0]["messages"][-1]["content"]

    # The recovered execution must not consume the original tool budget again.
    assert context.execution_budget_state.llm_calls == 2
    assert context.execution_budget_state.tool_calls == 1
    assert context.execution_budget_state.tool_rounds == 1

    assert len(checkpoint_handler.checkpoints) == 1
    recovered_checkpoint = checkpoint_handler.checkpoints[0]

    assert recovered_checkpoint.run_id == checkpoint.run_id
    assert recovered_checkpoint.agent_name == checkpoint.agent_name
    assert recovered_checkpoint.tool_round == 1
    assert recovered_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert recovered_checkpoint.messages[:3] == checkpoint.messages
    assert recovered_checkpoint.messages[3].role is AgentMessageRole.TOOL
    assert '"call_id": "call-recover-1"' in recovered_checkpoint.messages[3].content


@pytest.mark.asyncio
async def test_llm_agent_resume_restores_execution_budget_state() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=(),
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
            input="Continue the execution.",
            user_id="user-123",
            session_id="session-456",
            execution_budget=ExecutionBudget(
                max_llm_calls=1,
                max_tool_calls=5,
                max_tool_rounds=3,
                max_duration_seconds=300.0,
            ),
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-budget-resume-1",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-budget-resume-1",
        agent_name=definition.name,
        session_id="session-456",
        user_id="user-123",
        messages=(
            system_message("You are a production LLM agent."),
            user_message("Continue the execution."),
        ),
        tool_round=0,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
        execution_budget_state=ExecutionBudgetState(
            llm_calls=1,
            tool_calls=0,
            tool_rounds=0,
        ),
    )

    agent = LLMAgent(definition)

    with pytest.raises(
        AgentLLMCallLimitError,
        match="Agent 'production-llm-agent' exceeded the maximum LLM calls \\(1\\)",
    ):
        await agent.resume(context, checkpoint)

    assert len(gateway.requests) == 0
    assert context.execution_budget_state.llm_calls == 1
    assert context.execution_budget_state.tool_calls == 0
    assert context.execution_budget_state.tool_rounds == 0


@pytest.mark.asyncio
async def test_llm_agent_resume_continues_with_new_tool_call_and_checkpoint() -> None:
    definition = AgentDefinition(
        name="production-llm-agent",
        description="Production LLM agent.",
        system_prompt="You are a production LLM agent.",
        model="mock-gpt",
        tool_names=("search",),
    )

    gateway = FakeResumeContinuationLLMGateway()
    checkpoint_handler = FakeAgentCheckpointHandler()

    llm_context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model=definition.model,
            system_prompt=definition.system_prompt,
        ),
    )

    tool_registry = InMemoryToolRegistry()
    tool = FakeRAGTool(name="search")
    await tool_registry.register(tool)

    tools = AgentToolContext(
        tool_registry,
        definition,
    )

    context = AgentExecutionContext(
        AgentRequest(
            input="Find information about RAG.",
            user_id="user-123",
            session_id="session-456",
            metadata={"request_id": "request-789"},
        ),
        tools=tools,
        llm=llm_context,
        run_id="run-resume-2",
    )

    original_checkpoint = AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-resume-2",
        agent_name=definition.name,
        session_id="session-456",
        user_id="user-123",
        messages=(
            system_message("You are a production LLM agent."),
            user_message("Find information about RAG."),
            assistant_tool_call_message(
                tool_calls=(
                    AgentToolCall(
                        call_id="call-1",
                        name="search",
                        arguments={"query": "RAG"},
                    ),
                ),
                content="I searched for the initial information.",
            ),
            tool_result_message(
                call_id="call-1",
                tool_name="search",
                output={
                    "query": "RAG",
                    "retrieved_count": 2,
                },
            ),
        ),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={"request_id": "request-789"},
    )

    agent = LLMAgent(
        definition,
        checkpoint_handler=checkpoint_handler,
    )

    response = await agent.resume(context, original_checkpoint)

    assert response.output == "Resumed execution completed successfully."
    assert response.metadata["tool_rounds"] == 2

    assert tool.execute_count == 1
    assert len(gateway.requests) == 2

    assert gateway.requests[0]["messages"][-1] == {
        "role": "tool",
        "content": (
            '{"call_id": "call-1", '
            '"output": {"query": "RAG", "retrieved_count": 2}, '
            '"success": true, "tool_name": "search"}'
        ),
        "tool_call_id": "call-1",
        "tool_name": "search",
    }

    resumed_tool_result = gateway.requests[1]["messages"][-1]

    assert resumed_tool_result["role"] == "tool"
    assert resumed_tool_result["tool_call_id"] == "call-2"
    assert resumed_tool_result["tool_name"] == "search"
    assert '"call_id": "call-2"' in resumed_tool_result["content"]
    assert '"query": "follow-up"' in resumed_tool_result["content"]
    assert '"success": true' in resumed_tool_result["content"]

    assert len(checkpoint_handler.checkpoints) == 2

    before_checkpoint = checkpoint_handler.checkpoints[0]

    assert before_checkpoint.run_id == original_checkpoint.run_id
    assert before_checkpoint.agent_name == original_checkpoint.agent_name
    assert before_checkpoint.tool_round == 2
    assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
    assert before_checkpoint.metadata == original_checkpoint.metadata
    assert len(before_checkpoint.messages) == 5
    assert before_checkpoint.messages[:4] == original_checkpoint.messages
    assert before_checkpoint.messages[4] == assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="call-2",
                name="search",
                arguments={"query": "follow-up"},
            ),
        ),
        content="",
    )

    new_checkpoint = checkpoint_handler.checkpoints[1]

    assert new_checkpoint.run_id == original_checkpoint.run_id
    assert new_checkpoint.agent_name == original_checkpoint.agent_name
    assert new_checkpoint.tool_round == 2
    assert new_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
    assert new_checkpoint.metadata == original_checkpoint.metadata
    assert len(new_checkpoint.messages) == 6

    assert new_checkpoint.messages[:4] == original_checkpoint.messages
    assert new_checkpoint.messages[4] == assistant_tool_call_message(
        tool_calls=(
            AgentToolCall(
                call_id="call-2",
                name="search",
                arguments={"query": "follow-up"},
            ),
        ),
        content="",
    )
    assert new_checkpoint.messages[5].role is AgentMessageRole.TOOL
    assert '"call_id": "call-2"' in new_checkpoint.messages[5].content
    assert '"query": "follow-up"' in new_checkpoint.messages[5].content


@pytest.mark.asyncio
async def test_llm_agent_resume_rejects_mismatched_agent() -> None:
    context, _ = make_context()
    context = AgentExecutionContext(
        context.request,
        tools=context.tools,
        llm=context.llm,
        run_id="run-1",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-1",
        agent_name="different-agent",
        session_id=None,
        user_id=None,
        messages=(system_message("System."),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test agent.",
            system_prompt="System.",
            model="mock-gpt",
        )
    )

    with pytest.raises(
        ValueError,
        match="agent_name does not match",
    ):
        await agent.resume(context, checkpoint)


@pytest.mark.asyncio
async def test_llm_agent_resume_rejects_mismatched_run_id() -> None:
    context, _ = make_context()
    context = AgentExecutionContext(
        context.request,
        tools=context.tools,
        llm=context.llm,
        run_id="run-context",
    )

    checkpoint = AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-checkpoint",
        agent_name="test-llm-agent",
        session_id=None,
        user_id=None,
        messages=(system_message("System."),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )

    agent = LLMAgent(
        AgentDefinition(
            name="test-llm-agent",
            description="Test agent.",
            system_prompt="System.",
            model="mock-gpt",
        )
    )

    with pytest.raises(
        ValueError,
        match="run_id does not match",
    ):
        await agent.resume(context, checkpoint)
