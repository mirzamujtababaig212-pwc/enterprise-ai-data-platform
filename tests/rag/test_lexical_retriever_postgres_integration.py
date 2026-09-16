from __future__ import annotations

import asyncio
import os

import pytest
from sqlalchemy import create_engine, delete, text
from sqlalchemy.orm import Session, sessionmaker

from app.control_plane.persistence.models import RAGChunkRecord, RAGDocumentRecord
from rag.governance import GovernancePolicy
from rag.retrieval import PostgreSQLLexicalRetriever


def _postgres_connection():
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        engine.dispose()
        raise

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    return engine, session_factory


def _seed(
    session_factory,
    document_id: str,
) -> None:
    session: Session = session_factory()

    try:
        session.add(
            RAGDocumentRecord(
                document_id=document_id,
                content="PostgreSQL lexical retrieval integration test",
                document_metadata={"test": True},
            )
        )

        session.add_all(
            [
                RAGChunkRecord(
                    chunk_id=f"{document_id}:adjacent",
                    document_id=document_id,
                    chunk_index=0,
                    content="Gasoline engine diagnostics identify engine faults.",
                    chunk_metadata={"classification": "internal"},
                ),
                RAGChunkRecord(
                    chunk_id=f"{document_id}:separated",
                    document_id=document_id,
                    chunk_index=1,
                    content=(
                        "Gasoline vehicles require routine maintenance. "
                        "The engine supports combustion and diagnostics."
                    ),
                    chunk_metadata={"classification": "internal"},
                ),
                RAGChunkRecord(
                    chunk_id=f"{document_id}:restricted",
                    document_id=document_id,
                    chunk_index=2,
                    content="Gasoline engine maintenance requires inspection.",
                    chunk_metadata={"classification": "restricted"},
                ),
                RAGChunkRecord(
                    chunk_id=f"{document_id}:electric",
                    document_id=document_id,
                    chunk_index=3,
                    content="Electric vehicles use battery powered motors.",
                    chunk_metadata={"classification": "internal"},
                ),
            ]
        )

        session.commit()
    finally:
        session.close()


def _cleanup(
    session_factory,
    document_id: str,
) -> None:
    session: Session = session_factory()

    try:
        session.execute(delete(RAGChunkRecord).where(RAGChunkRecord.document_id == document_id))
        session.execute(
            delete(RAGDocumentRecord).where(RAGDocumentRecord.document_id == document_id)
        )
        session.commit()
    finally:
        session.close()


def test_postgresql_lexical_retriever_returns_ranked_matches() -> None:
    engine, session_factory = _postgres_connection()
    document_id = "lexical-postgres-ranked-matches"

    try:
        _seed(session_factory, document_id)

        retriever = PostgreSQLLexicalRetriever(
            session_factory=session_factory,
        )

        results = asyncio.run(
            retriever.retrieve(
                "gasoline engine",
                top_k=3,
            )
        )

        result_ids = {result.chunk.id for result in results}

        assert result_ids == {
            f"{document_id}:adjacent",
            f"{document_id}:separated",
            f"{document_id}:restricted",
        }
        assert results[0].chunk.id == f"{document_id}:adjacent"
        assert results[0].score == pytest.approx(1.0)
        assert all(0.0 <= result.score <= 1.0 for result in results)

    finally:
        _cleanup(session_factory, document_id)
        engine.dispose()


def test_postgresql_lexical_retriever_applies_top_k_and_min_score() -> None:
    engine, session_factory = _postgres_connection()
    document_id = "lexical-postgres-top-k-min-score"

    try:
        _seed(session_factory, document_id)

        retriever = PostgreSQLLexicalRetriever(
            session_factory=session_factory,
        )

        top_result = asyncio.run(
            retriever.retrieve(
                "gasoline engine",
                top_k=1,
            )
        )

        assert len(top_result) == 1
        assert top_result[0].chunk.id == f"{document_id}:adjacent"

        exact_score = asyncio.run(
            retriever.retrieve(
                "gasoline engine",
                top_k=5,
                min_score=1.0,
            )
        )

        assert exact_score
        assert all(result.score == pytest.approx(1.0) for result in exact_score)

    finally:
        _cleanup(session_factory, document_id)
        engine.dispose()


def test_postgresql_lexical_retriever_applies_metadata_filter() -> None:
    engine, session_factory = _postgres_connection()
    document_id = "lexical-postgres-metadata-filter"

    try:
        _seed(session_factory, document_id)

        retriever = PostgreSQLLexicalRetriever(
            session_factory=session_factory,
        )

        results = asyncio.run(
            retriever.retrieve(
                "gasoline engine",
                top_k=5,
                metadata_filter={"classification": "restricted"},
            )
        )

        assert [result.chunk.id for result in results] == [f"{document_id}:restricted"]

    finally:
        _cleanup(session_factory, document_id)
        engine.dispose()


def test_postgresql_lexical_retriever_applies_governance_policy() -> None:
    engine, session_factory = _postgres_connection()
    document_id = "lexical-postgres-governance"

    try:
        _seed(session_factory, document_id)

        retriever = PostgreSQLLexicalRetriever(
            session_factory=session_factory,
        )

        policy = GovernancePolicy(
            required_metadata={"classification": "internal"},
        )

        results = asyncio.run(
            retriever.retrieve(
                "gasoline engine",
                top_k=5,
                governance_policy=policy,
            )
        )

        result_ids = {result.chunk.id for result in results}

        assert f"{document_id}:restricted" not in result_ids
        assert f"{document_id}:adjacent" in result_ids
        assert f"{document_id}:separated" in result_ids

    finally:
        _cleanup(session_factory, document_id)
        engine.dispose()


def test_postgresql_lexical_retriever_rejects_conflicting_governance_policy() -> None:
    engine, session_factory = _postgres_connection()
    document_id = "lexical-postgres-governance-conflict"

    try:
        _seed(session_factory, document_id)

        retriever = PostgreSQLLexicalRetriever(
            session_factory=session_factory,
        )

        policy = GovernancePolicy(
            required_metadata={"classification": "internal"},
        )

        with pytest.raises(ValueError, match="conflicts with governance policy"):
            asyncio.run(
                retriever.retrieve(
                    "gasoline engine",
                    metadata_filter={"classification": "restricted"},
                    governance_policy=policy,
                )
            )

    finally:
        _cleanup(session_factory, document_id)
        engine.dispose()


def test_postgresql_lexical_retriever_returns_empty_for_no_match() -> None:
    engine, session_factory = _postgres_connection()
    document_id = "lexical-postgres-no-match"

    try:
        _seed(session_factory, document_id)

        retriever = PostgreSQLLexicalRetriever(
            session_factory=session_factory,
        )

        results = asyncio.run(
            retriever.retrieve(
                "quantum computing",
                top_k=5,
            )
        )

        assert results == []

    finally:
        _cleanup(session_factory, document_id)
        engine.dispose()


def test_postgresql_lexical_retriever_rejects_invalid_arguments() -> None:
    engine, session_factory = _postgres_connection()
    document_id = "lexical-postgres-validation"

    try:
        _seed(session_factory, document_id)

        retriever = PostgreSQLLexicalRetriever(
            session_factory=session_factory,
        )

        with pytest.raises(ValueError, match="empty"):
            asyncio.run(retriever.retrieve("   "))

        with pytest.raises(ValueError, match="top_k"):
            asyncio.run(retriever.retrieve("gasoline", top_k=0))

        with pytest.raises(ValueError, match="min_score"):
            asyncio.run(retriever.retrieve("gasoline", min_score=1.1))

    finally:
        _cleanup(session_factory, document_id)
        engine.dispose()
