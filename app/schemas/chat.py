from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)

    model: str | None = None

    provider: str | None = None

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    max_tokens: int | None = Field(
        default=None,
        gt=0,
    )


class ChatCompletionResponse(BaseModel):
    request_id: str
    provider: str
    model: str
    content: str

    usage: dict[str, int]

    metadata: dict[str, object] = {}
