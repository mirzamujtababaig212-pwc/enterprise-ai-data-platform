from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ai_platform.llm_gateway.config.settings import settings

from app.control_plane.dependencies import (
    get_rag_indexer,
    get_rag_query_service,
    get_rag_state_repository,
)
from app.control_plane.schemas.rag import (
    RAGIndexRequest,
    RAGIndexResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGSourceResponse,
)
from rag.indexing import RAGIndexer
from rag.models import Document
from rag.query import RAGQueryService
from app.control_plane.rag_state_repository import RAGStateRepository

router = APIRouter(
    prefix="/api/v1/rag",
    tags=["rag"],
)


@router.post(
    "/index",
    response_model=RAGIndexResponse,
)
async def index_document(
    request: RAGIndexRequest,
    indexer: RAGIndexer = Depends(get_rag_indexer),
    state_repository: RAGStateRepository = Depends(get_rag_state_repository),
) -> RAGIndexResponse:
    try:
        document = Document(
            id=request.document_id,
            content=request.content,
            metadata=request.metadata,
        )

        embedded_chunks = await indexer.index(document)

        chunks = [embedded_chunk.chunk for embedded_chunk in embedded_chunks]

        embedding_dimension = len(embedded_chunks[0].embedding) if embedded_chunks else None

        state_repository.save_document(
            document,
            chunks,
            embedding_model=settings.DEFAULT_EMBEDDING_MODEL,
            embedding_dimension=embedding_dimension,
        )

        return RAGIndexResponse(
            document_id=document.id,
            chunks_indexed=len(embedded_chunks),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.post(
    "/query",
    response_model=RAGQueryResponse,
)
async def query_rag(
    request: RAGQueryRequest,
    service: RAGQueryService = Depends(get_rag_query_service),
) -> RAGQueryResponse:
    try:
        result = await service.query(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            user_id=request.user_id,
        )

        return RAGQueryResponse(
            answer=result.answer,
            sources=[
                RAGSourceResponse(
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    score=source.score,
                    content=source.content,
                    metadata=source.metadata,
                )
                for source in result.sources
            ],
            retrieved_count=result.retrieved_count,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
