import pytest
from pydantic import ValidationError

from ai_platform.llm_gateway.models.chat import (
    ChatMessage,
    ChatRequest,
    ChatToolCall,
    ChatToolDefinition,
)


def test_chat_message_accepts_valid_message():
    message = ChatMessage(
        role="user",
        content="Explain RAG.",
    )

    assert message.role == "user"
    assert message.content == "Explain RAG."


def test_chat_message_rejects_empty_role():
    with pytest.raises(ValidationError):
        ChatMessage(
            role="",
            content="Explain RAG.",
        )


def test_chat_message_rejects_whitespace_role():
    with pytest.raises(ValidationError):
        ChatMessage(
            role="   ",
            content="Explain RAG.",
        )


def test_chat_message_rejects_empty_content():
    with pytest.raises(ValidationError):
        ChatMessage(
            role="user",
            content="",
        )


def test_chat_message_rejects_whitespace_content():
    with pytest.raises(ValidationError):
        ChatMessage(
            role="user",
            content="   ",
        )


def test_chat_request_accepts_prompt_without_messages():
    request = ChatRequest(
        prompt="Explain RAG.",
        model="gpt-4o",
    )

    assert request.prompt == "Explain RAG."
    assert request.messages is None


def test_chat_request_accepts_structured_messages():
    request = ChatRequest(
        prompt="Explain RAG.",
        model="gpt-4o",
        messages=[
            ChatMessage(
                role="system",
                content="You are an enterprise AI assistant.",
            ),
            ChatMessage(
                role="user",
                content="Explain RAG.",
            ),
        ],
    )

    assert request.messages is not None
    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"


def test_chat_request_accepts_message_dicts():
    request = ChatRequest(
        prompt="Explain RAG.",
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an enterprise AI assistant.",
            },
            {
                "role": "user",
                "content": "Explain RAG.",
            },
        ],
    )

    assert request.messages is not None
    assert request.messages[0].role == "system"
    assert request.messages[1].content == "Explain RAG."


def test_chat_request_rejects_invalid_message():
    with pytest.raises(ValidationError):
        ChatRequest(
            prompt="Explain RAG.",
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "",
                }
            ],
        )


def test_chat_tool_definition_accepts_valid_tool() -> None:
    tool = ChatToolDefinition(
        name="search",
        description="Search enterprise knowledge.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    )

    assert tool.name == "search"
    assert tool.description == "Search enterprise knowledge."
    assert tool.input_schema["type"] == "object"


def test_chat_request_accepts_multiple_tools() -> None:
    request = ChatRequest(
        prompt="Find the answer.",
        model="gpt-4o",
        tools=[
            ChatToolDefinition(
                name="search",
                description="Search knowledge.",
            ),
            ChatToolDefinition(
                name="calculator",
                description="Perform calculations.",
            ),
        ],
    )

    assert request.tools is not None
    assert [tool.name for tool in request.tools] == [
        "search",
        "calculator",
    ]


def test_chat_request_allows_empty_tools() -> None:
    request = ChatRequest(
        prompt="Hello.",
        model="gpt-4o",
        tools=[],
    )

    assert request.tools == []


def test_chat_tool_definition_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        ChatToolDefinition(
            name="",
            description="Search knowledge.",
        )


def test_chat_tool_definition_rejects_empty_description() -> None:
    with pytest.raises(ValueError):
        ChatToolDefinition(
            name="search",
            description="",
        )


def test_chat_tool_call_accepts_valid_tool_call() -> None:
    tool_call = ChatToolCall(
        call_id="call-123",
        name="search",
        arguments={
            "query": "RAG",
        },
    )

    assert tool_call.call_id == "call-123"
    assert tool_call.name == "search"
    assert tool_call.arguments == {
        "query": "RAG",
    }


def test_chat_message_accepts_tool_calls() -> None:
    message = ChatMessage(
        role="assistant",
        content="I will search the knowledge base.",
        tool_calls=[
            ChatToolCall(
                call_id="call-123",
                name="search",
                arguments={
                    "query": "RAG",
                },
            ),
        ],
    )

    assert message.role == "assistant"
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    assert message.tool_calls[0].call_id == "call-123"
    assert message.tool_calls[0].name == "search"


def test_chat_message_accepts_tool_result_metadata() -> None:
    message = ChatMessage(
        role="tool",
        content='{"status": "healthy"}',
        tool_call_id="call-123",
        tool_name="pipeline_status",
    )

    assert message.role == "tool"
    assert message.content == '{"status": "healthy"}'
    assert message.tool_call_id == "call-123"
    assert message.tool_name == "pipeline_status"


def test_chat_tool_call_rejects_empty_call_id() -> None:
    with pytest.raises(ValidationError):
        ChatToolCall(
            call_id="",
            name="search",
        )


def test_chat_tool_call_rejects_empty_tool_name() -> None:
    with pytest.raises(ValidationError):
        ChatToolCall(
            call_id="call-123",
            name=" ",
        )
