from pydantic import BaseModel, Field, field_validator
from typing import Any
from ai_platform.llm_gateway.models.usage import UsageMetrics


class ChatToolCall(BaseModel):
    call_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("call_id", "name")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chat tool call fields must not be empty.")
        return value


class ChatMessage(BaseModel):
    role: str
    content: str
    tool_calls: list[ChatToolCall] | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None

    @field_validator("role", "content")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chat message fields must not be empty.")
        return value

    @field_validator("tool_call_id", "tool_name")
    @classmethod
    def validate_optional_non_empty(
        cls,
        value: str | None,
    ) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Chat tool metadata fields must not be empty.")
        return value


class ChatToolDefinition(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    input_schema: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "description")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chat tool fields must not be empty.")
        return value


class ChatRequest(BaseModel):
    prompt: str = Field(
        json_schema_extra={
            "example": "Explain Retrieval-Augmented Generation.",
        }
    )

    messages: list[ChatMessage] | None = Field(
        default=None,
        json_schema_extra={
            "example": [
                {
                    "role": "system",
                    "content": "You are an enterprise AI assistant.",
                },
                {
                    "role": "user",
                    "content": "Explain Retrieval-Augmented Generation.",
                },
            ],
        },
    )

    tools: list[ChatToolDefinition] | None = Field(
        default=None,
        description="Optional tool definitions available to the model.",
    )

    provider: str = Field(
        default="openai",
        json_schema_extra={
            "example": "openai",
        },
    )

    model: str = Field(
        json_schema_extra={
            "example": "openai-gpt",
        }
    )

    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2,
        json_schema_extra={
            "example": 0.7,
        },
    )

    max_tokens: int = Field(
        default=1024,
        json_schema_extra={
            "example": 1024,
        },
    )

    stream: bool = Field(
        default=False,
        json_schema_extra={
            "example": False,
        },
    )

    user_id: str | None = Field(
        default=None,
        json_schema_extra={
            "example": "user123",
        },
    )


class ChatResponse(BaseModel):
    reply: str
    metrics: UsageMetrics
