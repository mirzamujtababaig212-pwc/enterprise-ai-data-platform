from pydantic import BaseModel, Field

from ai_platform.llm_gateway.models.usage import UsageMetrics


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: str = Field(
        json_schema_extra={
            "example": "Explain Retrieval-Augmented Generation.",
        }
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
