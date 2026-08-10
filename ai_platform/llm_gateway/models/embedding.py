from pydantic import BaseModel, Field

from ai_platform.llm_gateway.models.usage import UsageMetrics


class EmbeddingRequest(BaseModel):
    provider: str = Field(
        default="openai",
        json_schema_extra={
            "example": "openai",
        },
    )

    model: str = Field(
        json_schema_extra={
            "example": "openai-embedding",
        },
    )

    text: str = Field(
        json_schema_extra={
            "example": "Hello world",
        },
    )


class EmbeddingResponse(BaseModel):
    vector: list[float]
    metrics: UsageMetrics
