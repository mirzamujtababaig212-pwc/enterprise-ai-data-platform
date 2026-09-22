from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


def test_application_dependencies_use_postgres_memory_service() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    script = """
import asyncio

from app.control_plane.dependencies import (
    _memory_embedding_store,
    _memory_retriever,
    _memory_service,
    _memory_store,
)
from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore
from memory.retrieval.hybrid import HybridMemoryRetriever
from memory.retrieval.postgres_lexical import PostgreSQLLexicalMemoryRetriever
from memory.retrieval.postgres_semantic import PostgreSQLSemanticMemoryRetriever
from memory.stores.postgres import PostgreSQLMemoryStore


async def main() -> None:
    assert isinstance(_memory_store, PostgreSQLMemoryStore)
    assert isinstance(
        _memory_embedding_store,
        PostgreSQLMemoryEmbeddingStore,
    )
    assert _memory_service.embedding_service is not None
    assert _memory_service.embedding_store is _memory_embedding_store
    assert isinstance(_memory_retriever, HybridMemoryRetriever)
    assert isinstance(
        _memory_retriever.semantic_retriever,
        PostgreSQLSemanticMemoryRetriever,
    )
    assert isinstance(
        _memory_retriever.lexical_retriever,
        PostgreSQLLexicalMemoryRetriever,
    )

    namespace = "postgres-memory-application-integration"

    item = await _memory_service.remember(
        "Application-level PostgreSQL memory integration test.",
        namespace=namespace,
        memory_type="episodic",
        metadata={
            "source": "application-integration-test",
            "scope": "dependency-wiring",
        },
    )

    try:
        results = await _memory_service.recall(
            namespace,
            memory_type="episodic",
            limit=10,
        )
        retrieved = await _memory_retriever.retrieve(
            item.content,
            namespace=namespace,
            memory_type="episodic",
            top_k=5,
        )

        assert len(results) == 1

        result = results[0]

        assert result.id == item.id
        assert result.content == item.content
        assert result.namespace == namespace
        assert result.memory_type == "episodic"
        assert result.metadata == {
            "source": "application-integration-test",
            "scope": "dependency-wiring",
        }
        assert result.expires_at is None
        assert result.created_at.tzinfo is not None
        assert [result.item.id for result in retrieved] == [item.id]
    finally:
        await _memory_service.forget(item.id)


asyncio.run(main())
"""

    environment = os.environ.copy()
    environment["MEMORY_STORE_BACKEND"] = "postgres"
    environment["POSTGRES_HOST"] = "localhost"
    environment["POSTGRES_PORT"] = "5432"
    environment["DEFAULT_PROVIDER"] = "mock"
    environment["DEFAULT_EMBEDDING_MODEL"] = "mock-embedding"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "Application PostgreSQL memory integration failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
