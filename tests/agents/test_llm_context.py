from __future__ import annotations

from typing import Any

import pytest
from ai_platform.llm_gateway.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)
from ai_platform.agents.llm_config import AgentLLMConfig
from ai_platform.agents.llm_context import AgentLLMContext
from ai_platform.agents.llm_messages import (
    assistant_message,
    assistant_tool_call_message,
    system_message,
    tool_message,
    tool_result_message,
    user_message,
)
from ai_platform.agents.llm_result import (
    AgentLLMResult,
    AgentLLMUsage,
)
from ai_platform.agents.tool_calls import AgentToolCall
from tools.models import ToolDefinition


class FakeGateway:
    def __init__(
        self,
        response: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self.response = response
        self.error = error

    async def route_chat(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        self.requests.append(request)

        if self.error is not None:
            raise self.error

        if self.response is not None:
            return self.response

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


@pytest.mark.asyncio
async def test_generate_delegates_to_gateway() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    response = await context.generate(
        prompt="Explain RAG.",
        model="mock-gpt",
    )

    assert isinstance(response, AgentLLMResult)
    assert response.text == "Generated: Explain RAG."
    assert response.provider == "mock"
    assert response.model == "mock-gpt"

    assert gateway.requests == [
        {
            "prompt": "Explain RAG.",
            "model": "mock-gpt",
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False,
        }
    ]


@pytest.mark.asyncio
async def test_generate_passes_structured_messages_to_gateway() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    messages = (
        system_message("You are an agent."),
        user_message("Explain RAG."),
    )

    await context.generate(
        prompt="Explain RAG.",
        messages=messages,
        model="mock-gpt",
    )

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "You are an agent.",
        },
        {
            "role": "user",
            "content": "Explain RAG.",
        },
    ]


@pytest.mark.asyncio
async def test_generate_preserves_structured_message_order() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    messages = (
        system_message("System instruction."),
        user_message("First question."),
        assistant_message("First answer."),
        user_message("Follow-up question."),
        tool_message("Tool result."),
    )

    await context.generate(
        prompt="Follow-up question.",
        messages=messages,
        model="mock-gpt",
    )

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "System instruction.",
        },
        {
            "role": "user",
            "content": "First question.",
        },
        {
            "role": "assistant",
            "content": "First answer.",
        },
        {
            "role": "user",
            "content": "Follow-up question.",
        },
        {
            "role": "tool",
            "content": "Tool result.",
        },
    ]


@pytest.mark.asyncio
async def test_generate_rejects_invalid_structured_message() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        TypeError,
        match="LLM messages must contain AgentMessage instances",
    ):
        await context.generate(
            prompt="Hello.",
            messages=(
                system_message("You are an agent."),
                "invalid-message",  # type: ignore[arg-type]
            ),
            model="mock-gpt",
        )


@pytest.mark.asyncio
async def test_generate_omits_messages_for_legacy_prompt_only_call() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    await context.generate(
        prompt="Legacy prompt.",
        model="mock-gpt",
    )

    assert "messages" not in gateway.requests[0]


@pytest.mark.asyncio
async def test_generate_passes_provider() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    await context.generate(
        prompt="Hello.",
        model="mock-gpt",
        provider="mock",
    )

    assert gateway.requests[0]["provider"] == "mock"


@pytest.mark.asyncio
async def test_generate_passes_generation_parameters() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    await context.generate(
        prompt="Summarize this.",
        model="mock-gpt",
        temperature=0.2,
        max_tokens=256,
    )

    request = gateway.requests[0]

    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 256
    assert request["stream"] is False


@pytest.mark.asyncio
async def test_generate_passes_user_id() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    await context.generate(
        prompt="Hello.",
        model="mock-gpt",
        user_id="user-123",
    )

    assert gateway.requests[0]["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_generate_preserves_gateway_response() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    response = await context.generate(
        prompt="Hello.",
        model="mock-gpt",
    )

    assert isinstance(response, AgentLLMResult)
    assert response.text == "Generated: Hello."
    assert response.provider == "mock"
    assert response.model == "mock-gpt"
    assert response.usage == AgentLLMUsage(
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
    )
    assert response.raw_response == {
        "provider": "mock",
        "model": "mock-gpt",
        "reply": "Generated: Hello.",
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }


@pytest.mark.asyncio
async def test_generate_rejects_empty_prompt() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="LLM prompt must not be empty",
    ):
        await context.generate(
            prompt="",
            model="mock-gpt",
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_model() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="LLM model must be provided either by the agent configuration or the generate call.",
    ):
        await context.generate(
            prompt="Hello.",
            model="",
        )


@pytest.mark.asyncio
async def test_generate_rejects_invalid_temperature() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="temperature must be between 0 and 2",
    ):
        await context.generate(
            prompt="Hello.",
            model="mock-gpt",
            temperature=2.1,
        )


@pytest.mark.asyncio
async def test_generate_rejects_invalid_max_tokens() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="max_tokens must be greater than zero",
    ):
        await context.generate(
            prompt="Hello.",
            model="mock-gpt",
            max_tokens=0,
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_provider() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="LLM provider must not be empty",
    ):
        await context.generate(
            prompt="Hello.",
            model="mock-gpt",
            provider="   ",
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_user_id() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="LLM user_id must not be empty",
    ):
        await context.generate(
            prompt="Hello.",
            model="mock-gpt",
            user_id="   ",
        )


@pytest.mark.asyncio
async def test_llm_context_exposes_bound_agent_configuration() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="bound-model",
            system_prompt="You are a bound agent.",
            temperature=0.3,
            max_tokens=512,
        ),
    )

    assert context.model == "bound-model"
    assert context.system_prompt == "You are a bound agent."
    assert context.temperature == 0.3
    assert context.max_tokens == 512


@pytest.mark.asyncio
async def test_generate_uses_bound_model_when_model_is_omitted() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="bound-model",
            system_prompt="You are a bound agent.",
        ),
    )

    await context.generate(
        prompt="Hello.",
    )

    assert gateway.requests[0]["model"] == "bound-model"


@pytest.mark.asyncio
async def test_generate_uses_bound_generation_defaults() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="bound-model",
            system_prompt="You are a bound agent.",
            temperature=0.25,
            max_tokens=512,
        ),
    )

    await context.generate(
        prompt="Hello.",
    )

    assert gateway.requests[0]["temperature"] == 0.25
    assert gateway.requests[0]["max_tokens"] == 512


@pytest.mark.asyncio
async def test_generate_allows_explicit_model_override() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="bound-model",
            system_prompt="You are a bound agent.",
        ),
    )

    await context.generate(
        prompt="Hello.",
        model="override-model",
    )

    assert gateway.requests[0]["model"] == "override-model"


@pytest.mark.asyncio
async def test_build_messages_places_system_prompt_first() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an enterprise agent.",
        ),
    )

    messages = context.build_messages(
        prompt="Explain RAG.",
    )

    assert messages == (
        system_message("You are an enterprise agent."),
        user_message("Explain RAG."),
    )


@pytest.mark.asyncio
async def test_build_messages_preserves_conversation_history() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an enterprise agent.",
        ),
    )

    history = (
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
    )

    messages = context.build_messages(
        prompt="Why is retrieval useful?",
        history=history,
    )

    assert messages == (
        system_message("You are an enterprise agent."),
        user_message("What is RAG?"),
        assistant_message("RAG retrieves relevant context."),
        user_message("Why is retrieval useful?"),
    )


@pytest.mark.asyncio
async def test_build_messages_appends_tool_results() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an enterprise agent.",
        ),
    )

    messages = context.build_messages(
        prompt="What is the current status?",
        tool_results=(
            '{"status": "healthy"}',
            '{"latency_ms": 42}',
        ),
    )

    assert messages == (
        system_message("You are an enterprise agent."),
        user_message("What is the current status?"),
        tool_message('{"status": "healthy"}'),
        tool_message('{"latency_ms": 42}'),
    )


@pytest.mark.asyncio
async def test_build_messages_rejects_empty_prompt() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an enterprise agent.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="LLM prompt must not be empty",
    ):
        context.build_messages(
            prompt="   ",
        )


@pytest.mark.asyncio
async def test_build_messages_rejects_invalid_history_message() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an enterprise agent.",
        ),
    )

    with pytest.raises(
        TypeError,
        match="LLM history must contain AgentMessage instances",
    ):
        context.build_messages(
            prompt="Hello.",
            history=("invalid",),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_generate_normalizes_gateway_response() -> None:
    gateway = FakeGateway(
        response={
            "provider": "openai",
            "model": "gpt-test",
            "reply": "Generated answer.",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
    )

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="gpt-test",
            system_prompt="You are an agent.",
        ),
    )

    result = await context.generate(
        prompt="Hello.",
        model="gpt-test",
    )

    assert isinstance(result, AgentLLMResult)
    assert result.text == "Generated answer."
    assert result.provider == "openai"
    assert result.model == "gpt-test"
    assert result.usage == AgentLLMUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )


@pytest.mark.asyncio
async def test_generate_preserves_raw_gateway_response() -> None:
    gateway_response = {
        "provider": "openai",
        "model": "gpt-test",
        "reply": "Generated answer.",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "request_id": "req-123",
    }

    gateway = FakeGateway(
        response=gateway_response,
    )

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="gpt-test",
            system_prompt="You are an agent.",
        ),
    )

    result = await context.generate(
        prompt="Hello.",
        model="gpt-test",
    )

    assert result.raw_response == gateway_response


@pytest.mark.asyncio
async def test_generate_propagates_authentication_error_unchanged() -> None:
    gateway_error = AuthenticationError(
        "Invalid API key.",
        provider="openai",
        model="gpt-test",
    )

    gateway = FakeGateway(error=gateway_error)

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="gpt-test",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(AuthenticationError) as exc_info:
        await context.generate(
            prompt="Hello.",
            model="gpt-test",
        )

    assert exc_info.value is gateway_error


@pytest.mark.asyncio
async def test_generate_propagates_rate_limit_error_unchanged() -> None:
    gateway_error = RateLimitError(
        "Rate limit exceeded.",
        provider="openai",
        model="gpt-test",
        retry_after_seconds=30.0,
    )

    gateway = FakeGateway(error=gateway_error)

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="gpt-test",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(RateLimitError) as exc_info:
        await context.generate(
            prompt="Hello.",
            model="gpt-test",
        )

    assert exc_info.value is gateway_error
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after_seconds == 30.0


@pytest.mark.asyncio
async def test_generate_propagates_provider_unavailable_error_unchanged() -> None:
    gateway_error = ProviderUnavailableError(
        "Provider unavailable.",
        provider="openai",
        model="gpt-test",
    )

    gateway = FakeGateway(error=gateway_error)

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="gpt-test",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(ProviderUnavailableError) as exc_info:
        await context.generate(
            prompt="Hello.",
            model="gpt-test",
        )

    assert exc_info.value is gateway_error
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_generate_propagates_model_not_found_error_unchanged() -> None:
    gateway_error = ModelNotFoundError(
        "Model not found.",
        provider="openai",
        model="gpt-missing",
    )

    gateway = FakeGateway(error=gateway_error)

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="gpt-missing",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(ModelNotFoundError) as exc_info:
        await context.generate(
            prompt="Hello.",
            model="gpt-missing",
        )

    assert exc_info.value is gateway_error


@pytest.mark.asyncio
async def test_generate_propagates_tool_calls() -> None:
    tool_call = AgentToolCall(
        call_id="call-1",
        name="search_documents",
        arguments={
            "query": "enterprise AI",
        },
    )

    gateway = FakeGateway(
        response={
            "provider": "mock",
            "model": "mock-gpt",
            "reply": "I need to search the documents.",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "tool_calls": [tool_call],
        }
    )

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    response = await context.generate(
        prompt="Find the relevant documents.",
        model="mock-gpt",
    )

    assert response.tool_calls == (tool_call,)
    assert response.tool_calls[0].call_id == "call-1"
    assert response.tool_calls[0].name == "search_documents"
    assert response.tool_calls[0].arguments == {
        "query": "enterprise AI",
    }


@pytest.mark.asyncio
async def test_generate_preserves_multiple_tool_calls() -> None:
    first = AgentToolCall(
        call_id="call-1",
        name="search_documents",
        arguments={
            "query": "RAG",
        },
    )

    second = AgentToolCall(
        call_id="call-2",
        name="get_document",
        arguments={
            "document_id": "doc-123",
        },
    )

    gateway = FakeGateway(
        response={
            "provider": "mock",
            "model": "mock-gpt",
            "reply": "I need to retrieve information.",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18,
            },
            "tool_calls": [first, second],
        }
    )

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    response = await context.generate(
        prompt="Search and retrieve the document.",
        model="mock-gpt",
    )

    assert response.tool_calls == (
        first,
        second,
    )


@pytest.mark.asyncio
async def test_generate_defaults_missing_tool_calls_to_empty_tuple() -> None:
    gateway = FakeGateway(
        response={
            "provider": "mock",
            "model": "mock-gpt",
            "reply": "Normal response.",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        }
    )

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    response = await context.generate(
        prompt="Hello.",
        model="mock-gpt",
    )

    assert response.tool_calls == ()


@pytest.mark.asyncio
async def test_generate_rejects_invalid_tool_call_objects() -> None:
    gateway = FakeGateway(
        response={
            "provider": "mock",
            "model": "mock-gpt",
            "reply": "Invalid tool call response.",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
            "tool_calls": [
                {
                    "call_id": "call-1",
                    "name": "search_documents",
                    "arguments": {},
                }
            ],
        }
    )

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        TypeError,
        match="tool_calls must contain AgentToolCall instances",
    ):
        await context.generate(
            prompt="Hello.",
            model="mock-gpt",
        )


@pytest.mark.asyncio
async def test_generate_rejects_invalid_tool_calls_container() -> None:
    gateway = FakeGateway(
        response={
            "provider": "mock",
            "model": "mock-gpt",
            "reply": "Invalid tool calls container.",
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
            "tool_calls": "not-a-list",
        }
    )

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        ValueError,
        match="tool_calls must be a list or tuple",
    ):
        await context.generate(
            prompt="Hello.",
            model="mock-gpt",
        )


@pytest.mark.asyncio
async def test_generate_passes_tool_definitions_to_gateway() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    tools = (
        ToolDefinition(
            name="search",
            description="Search enterprise knowledge.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                },
                "required": ["query"],
            },
        ),
    )

    await context.generate(
        prompt="Find information about RAG.",
        model="mock-gpt",
        tools=tools,
    )

    assert gateway.requests[0]["tools"] == [
        {
            "name": "search",
            "description": "Search enterprise knowledge.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                    },
                },
                "required": ["query"],
            },
        },
    ]


@pytest.mark.asyncio
async def test_generate_preserves_multiple_tool_definitions_order() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    tools = (
        ToolDefinition(
            name="search",
            description="Search knowledge.",
        ),
        ToolDefinition(
            name="calculator",
            description="Perform calculations.",
        ),
    )

    await context.generate(
        prompt="Use the available tools.",
        model="mock-gpt",
        tools=tools,
    )

    assert [tool["name"] for tool in gateway.requests[0]["tools"]] == [
        "search",
        "calculator",
    ]


@pytest.mark.asyncio
async def test_generate_allows_empty_tool_definitions() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    await context.generate(
        prompt="Hello.",
        model="mock-gpt",
        tools=(),
    )

    assert gateway.requests[0]["tools"] == []


@pytest.mark.asyncio
async def test_generate_rejects_invalid_tool_definition() -> None:
    gateway = FakeGateway()

    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    with pytest.raises(
        TypeError,
        match="LLM tools must contain ToolDefinition instances",
    ):
        await context.generate(
            prompt="Hello.",
            model="mock-gpt",
            tools=("invalid-tool",),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_generate_serializes_ordinary_messages() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    messages = (
        system_message("System instruction."),
        user_message("Hello."),
        assistant_message("Hi there."),
    )

    await context.generate(
        prompt="Continue.",
        messages=messages,
        model="mock-gpt",
    )

    assert gateway.requests[0]["messages"] == [
        {
            "role": "system",
            "content": "System instruction.",
        },
        {
            "role": "user",
            "content": "Hello.",
        },
        {
            "role": "assistant",
            "content": "Hi there.",
        },
    ]


@pytest.mark.asyncio
async def test_generate_serializes_assistant_tool_call() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    messages = (
        system_message("You are an agent."),
        user_message("Search for RAG."),
        assistant_tool_call_message(
            tool_calls=(
                AgentToolCall(
                    call_id="call-123",
                    name="search",
                    arguments={"query": "RAG"},
                ),
            ),
        ),
    )

    await context.generate(
        prompt="Search for RAG.",
        messages=messages,
        model="mock-gpt",
    )

    assistant = gateway.requests[0]["messages"][2]

    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Tool call requested."
    assert assistant["tool_calls"] == [
        {
            "call_id": "call-123",
            "name": "search",
            "arguments": {"query": "RAG"},
        }
    ]


@pytest.mark.asyncio
async def test_generate_preserves_multiple_assistant_tool_calls() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    messages = (
        assistant_tool_call_message(
            tool_calls=(
                AgentToolCall(
                    call_id="call-1",
                    name="search",
                    arguments={"query": "RAG"},
                ),
                AgentToolCall(
                    call_id="call-2",
                    name="status",
                    arguments={"service": "gateway"},
                ),
            ),
        ),
    )

    await context.generate(
        prompt="Run both tools.",
        messages=messages,
        model="mock-gpt",
    )

    assert gateway.requests[0]["messages"][0]["tool_calls"] == [
        {
            "call_id": "call-1",
            "name": "search",
            "arguments": {"query": "RAG"},
        },
        {
            "call_id": "call-2",
            "name": "status",
            "arguments": {"service": "gateway"},
        },
    ]


@pytest.mark.asyncio
async def test_generate_serializes_tool_result_metadata() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    messages = (
        user_message("Check the gateway."),
        tool_result_message(
            call_id="call-123",
            tool_name="gateway_status",
            output={"status": "healthy"},
        ),
    )

    await context.generate(
        prompt="Check the gateway.",
        messages=messages,
        model="mock-gpt",
    )

    tool = gateway.requests[0]["messages"][1]

    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call-123"
    assert tool["tool_name"] == "gateway_status"
    assert '"success": true' in tool["content"]


@pytest.mark.asyncio
async def test_generate_serializes_tool_error_metadata() -> None:
    gateway = FakeGateway()
    context = AgentLLMContext(
        gateway,
        AgentLLMConfig(
            model="mock-gpt",
            system_prompt="You are an agent.",
        ),
    )

    messages = (
        user_message("Check the gateway."),
        tool_result_message(
            call_id="call-456",
            tool_name="gateway_status",
            error="Gateway unavailable.",
        ),
    )

    await context.generate(
        prompt="Check the gateway.",
        messages=messages,
        model="mock-gpt",
    )

    tool = gateway.requests[0]["messages"][1]

    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call-456"
    assert tool["tool_name"] == "gateway_status"
    assert '"success": false' in tool["content"]
    assert "Gateway unavailable." in tool["content"]


def test_serialize_messages_rejects_malformed_assistant_tool_calls() -> None:
    malformed = assistant_message('{"tool_calls": "not-a-list"}')

    with pytest.raises(
        ValueError,
        match="tool_calls must be a list",
    ):
        AgentLLMContext._serialize_messages((malformed,))


def test_serialize_messages_rejects_malformed_tool_result() -> None:
    malformed = tool_message('{"call_id": "", "tool_name": "search", "success": true}')

    with pytest.raises(
        ValueError,
        match="call_id must be a non-empty string",
    ):
        AgentLLMContext._serialize_messages((malformed,))


def test_normalize_response_allows_tool_call_only_response() -> None:
    tool_call = AgentToolCall(
        call_id="call-123",
        name="search",
        arguments={"query": "RAG"},
    )

    response = {
        "reply": "",
        "provider": "openai",
        "model": "gpt-test",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "tool_calls": [tool_call],
    }

    result = AgentLLMContext._normalize_response(response)

    assert result.text == ""
    assert result.provider == "openai"
    assert result.model == "gpt-test"
    assert result.tool_calls == (tool_call,)


def test_normalize_response_rejects_empty_reply_without_tool_calls() -> None:
    response = {
        "reply": "",
        "provider": "openai",
        "model": "gpt-test",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "tool_calls": [],
    }

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        AgentLLMContext._normalize_response(response)
