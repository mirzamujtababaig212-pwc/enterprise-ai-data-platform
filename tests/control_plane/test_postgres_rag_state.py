from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import Base
from app.control_plane.persistence.rag_state import (
    PostgreSQLRAGStateRepository,
)
from rag.models import Document, DocumentChunk


def build_repository() -> PostgreSQLRAGStateRepository:
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return PostgreSQLRAGStateRepository(session_factory())


def test_save_and_get_document() -> None:
    repository = build_repository()

    document = Document(
        id="doc-1",
        content="Enterprise AI Platform",
        metadata={
            "source": "architecture.md",
            "department": "engineering",
        },
    )

    chunks = [
        DocumentChunk(
            id="doc-1:chunk:0",
            document_id="doc-1",
            content="Enterprise AI",
            metadata={"source": "architecture.md"},
            chunk_index=0,
        ),
        DocumentChunk(
            id="doc-1:chunk:1",
            document_id="doc-1",
            content="Platform",
            metadata={"source": "architecture.md"},
            chunk_index=1,
        ),
    ]

    repository.save_document(
        document,
        chunks,
        embedding_model="test-embedding",
        embedding_dimension=2,
    )

    stored_document = repository.get_document("doc-1")
    stored_chunks = repository.get_chunks("doc-1")

    assert stored_document == document
    assert stored_chunks == chunks


def test_get_missing_document_returns_none() -> None:
    repository = build_repository()

    assert repository.get_document("missing") is None
    assert repository.get_chunks("missing") == []


def test_reindex_replaces_existing_chunks() -> None:
    repository = build_repository()

    document = Document(
        id="doc-1",
        content="Original content",
        metadata={"version": 1},
    )

    original_chunks = [
        DocumentChunk(
            id="doc-1:chunk:0",
            document_id="doc-1",
            content="Original chunk",
            metadata={"version": 1},
            chunk_index=0,
        ),
        DocumentChunk(
            id="doc-1:chunk:1",
            document_id="doc-1",
            content="Old second chunk",
            metadata={"version": 1},
            chunk_index=1,
        ),
    ]

    repository.save_document(document, original_chunks)

    updated_document = Document(
        id="doc-1",
        content="Updated content",
        metadata={"version": 2},
    )

    updated_chunks = [
        DocumentChunk(
            id="doc-1:chunk:0",
            document_id="doc-1",
            content="Updated chunk",
            metadata={"version": 2},
            chunk_index=0,
        ),
    ]

    repository.save_document(updated_document, updated_chunks)

    assert repository.get_document("doc-1") == updated_document
    assert repository.get_chunks("doc-1") == updated_chunks


def test_delete_document_removes_document_and_chunks() -> None:
    repository = build_repository()

    document = Document(
        id="doc-1",
        content="Content",
        metadata={"source": "test"},
    )

    chunks = [
        DocumentChunk(
            id="doc-1:chunk:0",
            document_id="doc-1",
            content="Content",
            metadata={"source": "test"},
            chunk_index=0,
        )
    ]

    repository.save_document(document, chunks)

    repository.delete_document("doc-1")

    assert repository.get_document("doc-1") is None
    assert repository.get_chunks("doc-1") == []


def test_save_document_rolls_back_on_commit_failure() -> None:
    repository = build_repository()

    document = Document(
        id="doc-rollback",
        content="Rollback content",
        metadata={"test": True},
    )

    repository._session.commit = lambda: (_ for _ in ()).throw(RuntimeError("commit failed"))

    try:
        repository.save_document(document, [])
    except RuntimeError as exc:
        assert str(exc) == "commit failed"
    else:
        raise AssertionError("Expected RuntimeError")


def test_delete_document_rolls_back_on_commit_failure() -> None:
    repository = build_repository()

    document = Document(
        id="doc-delete-rollback",
        content="Delete rollback content",
        metadata={"test": True},
    )

    chunks = [
        DocumentChunk(
            id="doc-delete-rollback:chunk:0",
            document_id="doc-delete-rollback",
            content="Chunk",
            metadata={},
            chunk_index=0,
        )
    ]

    repository.save_document(document, chunks)

    repository._session.commit = lambda: (_ for _ in ()).throw(RuntimeError("commit failed"))

    try:
        repository.delete_document(document.id)
    except RuntimeError as exc:
        assert str(exc) == "commit failed"
    else:
        raise AssertionError("Expected RuntimeError")
