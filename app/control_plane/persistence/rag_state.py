from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import (
    RAGChunkRecord,
    RAGDocumentRecord,
)
from rag.models import Document, DocumentChunk


class PostgreSQLRAGStateRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_document(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        *,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> None:
        try:
            existing = self._session.scalar(
                select(RAGDocumentRecord).where(RAGDocumentRecord.document_id == document.id)
            )

            if existing is not None:
                existing.content = document.content
                existing.document_metadata = dict(document.metadata)

                self._session.execute(
                    delete(RAGChunkRecord).where(RAGChunkRecord.document_id == document.id)
                )
            else:
                existing = RAGDocumentRecord(
                    document_id=document.id,
                    content=document.content,
                    document_metadata=dict(document.metadata),
                )
                self._session.add(existing)

            for chunk in chunks:
                self._session.add(
                    RAGChunkRecord(
                        chunk_id=chunk.id,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        chunk_metadata=dict(chunk.metadata),
                        embedding_model=embedding_model,
                        embedding_dimension=embedding_dimension,
                    )
                )

            self._session.commit()

        except Exception:
            self._session.rollback()
            raise

    def get_document(self, document_id: str) -> Document | None:
        record = self._session.scalar(
            select(RAGDocumentRecord).where(RAGDocumentRecord.document_id == document_id)
        )

        if record is None:
            return None

        return Document(
            id=record.document_id,
            content=record.content,
            metadata=dict(record.document_metadata),
        )

    def get_chunks(self, document_id: str) -> list[DocumentChunk]:
        statement = (
            select(RAGChunkRecord)
            .where(RAGChunkRecord.document_id == document_id)
            .order_by(RAGChunkRecord.chunk_index.asc())
        )

        records = list(self._session.scalars(statement).all())

        return [
            DocumentChunk(
                id=record.chunk_id,
                document_id=record.document_id,
                content=record.content,
                metadata=dict(record.chunk_metadata),
                chunk_index=record.chunk_index,
            )
            for record in records
        ]

    def delete_document(self, document_id: str) -> None:
        try:
            self._session.execute(
                delete(RAGChunkRecord).where(RAGChunkRecord.document_id == document_id)
            )
            self._session.execute(
                delete(RAGDocumentRecord).where(RAGDocumentRecord.document_id == document_id)
            )
            self._session.commit()

        except Exception:
            self._session.rollback()
            raise
