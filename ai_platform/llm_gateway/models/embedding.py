from pydantic import BaseModel
from ai_platform.llm_gateway.models.usage import UsageMetrics


class EmbeddingRequest(BaseModel):
    text: str
    provider: str = "openai"
    model: str


class EmbeddingResponse(BaseModel):
    vector: list[float]
    metrics: UsageMetrics
