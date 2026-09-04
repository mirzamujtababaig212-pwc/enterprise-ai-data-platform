from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RAGIndexRequest(BaseModel):
    document_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGIndexResponse(BaseModel):
    document_id: str
    chunks_indexed: int


class RAGQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0)
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0)
    user_id: str | None = None


class RAGSourceResponse(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    content: str
    metadata: dict[str, Any]


class RAGQueryResponse(BaseModel):
    answer: str
    sources: list[RAGSourceResponse]
    retrieved_count: int
