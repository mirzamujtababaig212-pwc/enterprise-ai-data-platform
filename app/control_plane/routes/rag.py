from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from app.control_plane.dependencies import (
    get_rag_indexer,
    get_rag_query_service,
    get_rag_state_repository,
    get_rag_vector_store,
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
from rag.contracts import VectorStore
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

        existing_chunks = state_repository.get_chunks(document.id)

        state_repository.ensure_document(document)

        embedded_chunks = await indexer.index(
            document,
            previous_chunk_ids=[chunk.id for chunk in existing_chunks],
        )

        state_repository.save_indexed_document(
            document,
            list(embedded_chunks),
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


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_document(
    document_id: str,
    vector_store: VectorStore = Depends(get_rag_vector_store),
    state_repository: RAGStateRepository = Depends(get_rag_state_repository),
) -> Response:
    try:
        existing_chunks = state_repository.get_chunks(document_id)
        chunk_ids = [chunk.id for chunk in existing_chunks]

        if chunk_ids:
            await vector_store.delete_chunks(chunk_ids)

        state_repository.delete_document(document_id)

        return Response(status_code=status.HTTP_204_NO_CONTENT)

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
    request: Request,
    payload: RAGQueryRequest,
    service: RAGQueryService = Depends(get_rag_query_service),
) -> RAGQueryResponse:
    try:
        result = await service.query(
            query=payload.query,
            top_k=payload.top_k,
            min_score=payload.min_score,
            metadata_filter=payload.metadata_filter,
            governance_policy=None,
            tenant_id=getattr(request.state, "tenant_id", None),
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            user_id=payload.user_id,
        )

        return RAGQueryResponse(
            answer=result.answer,
            sources=[
                RAGSourceResponse(
                    evidence_id=source.evidence_id,
                    chunk_id=source.chunk_id,
                    document_id=source.document_id,
                    score=source.score,
                    retrieval_rank=source.retrieval_rank,
                    content=source.content,
                    metadata=source.metadata,
                    source_ref=source.source_ref,
                    locator=source.locator,
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
