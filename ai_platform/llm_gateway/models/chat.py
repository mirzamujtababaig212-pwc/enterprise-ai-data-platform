from pydantic import BaseModel, Field
from typing import Optional
from ai_platform.llm_gateway.metrics.metrics import Metrics


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: str = Field(example="Explain Retrieval-Augmented Generation.")
    provider: str = Field(
        default="openai",
        example="openai",
    )
    model: str = Field(
        example="openai-gpt",
    )
    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2,
        example=0.7,
    )
    max_tokens: int = Field(
        default=1024,
        example=1024,
    )
    stream: bool = Field(
        default=False,
        example=False,
    )
    user_id: Optional[str] = Field(
        default=None,
        example="user123",
    )


class ChatResponse(BaseModel):
    reply: str
    metrics: Metrics
