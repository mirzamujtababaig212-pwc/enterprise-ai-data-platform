from __future__ import annotations

from typing import Protocol

from rag.models import Document, DocumentChunk


class RAGStateRepository(Protocol):
    def save_document(
        self,
        document: Document,
        chunks: list[DocumentChunk],
        *,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
    ) -> None: ...

    def get_document(self, document_id: str) -> Document | None: ...

    def get_chunks(self, document_id: str) -> list[DocumentChunk]: ...

    def delete_document(self, document_id: str) -> None: ...
