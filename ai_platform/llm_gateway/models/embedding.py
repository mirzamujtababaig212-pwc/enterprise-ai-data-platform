from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    text: str
    model: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    provider: str
    model: str
