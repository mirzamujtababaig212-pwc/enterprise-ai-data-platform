from pydantic import BaseModel, Field
from typing import Optional


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    prompt: str
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    stream: bool = False
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    provider: str
    model: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
