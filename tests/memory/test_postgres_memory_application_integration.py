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


def test_runtime_round_trip_uses_durable_postgres_memory() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    script = """
import asyncio
import os

from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.models import AgentDefinition, AgentRequest, AgentResponse
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from app.control_plane.persistence.models import MemoryItemRecord
from memory.context.builder import MemoryContextBuilder
from memory.service import MemoryService
from memory.stores.postgres import PostgreSQLMemoryStore


class MemoryWritingAgent:
    def __init__(self) -> None:
        self._definition = AgentDefinition(
            name="postgres-memory-round-trip-agent",
            description="PostgreSQL memory round-trip integration agent.",
            system_prompt="You are a deterministic memory integration test agent.",
            model="test-model",
            memory_write_enabled=True,
        )
        self.last_context: AgentExecutionContext | None = None

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        self.last_context = context

        return AgentResponse(
            agent_name=self.definition.name,
            output="Fleet-42 completed its recovery deployment successfully.",
            session_id=context.session_id,
        )


class MemoryReadingAgent:
    def __init__(self) -> None:
        self._definition = AgentDefinition(
            name="postgres-memory-reading-agent",
            description="PostgreSQL memory retrieval integration agent.",
            system_prompt="You are a deterministic memory retrieval integration test agent.",
            model="test-model",
        )
        self.last_context: AgentExecutionContext | None = None
        self.last_messages = ()

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(
        self,
        context: AgentExecutionContext,
    ) -> AgentResponse:
        self.last_context = context
        self.last_messages = context.build_llm_messages()

        return AgentResponse(
            agent_name=self.definition.name,
            output="Memory retrieved successfully.",
            session_id=context.session_id,
        )


async def main() -> None:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DATABASE", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
    )
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    namespace = "postgres-runtime-memory-round-trip"
    first_agent = MemoryWritingAgent()
    second_agent = MemoryReadingAgent()
    registry = InMemoryAgentRegistry()

    await registry.register(first_agent)
    await registry.register(second_agent)

    memory_service = MemoryService(
        PostgreSQLMemoryStore(session_factory),
    )
    memory_context_builder = MemoryContextBuilder(
        memory_service,
    )

    runtime = AgentRuntime(
        registry,
        memory_context_builder=memory_context_builder,
        memory_service=memory_service,
    )

    created_item = None

    try:
        first_response = await runtime.run(
            first_agent.definition.name,
            AgentRequest(
                input="Record the fleet-42 recovery result.",
                session_id="postgres-memory-session-1",
                memory_namespace=namespace,
            ),
        )

        assert (
            first_response.output
            == "Fleet-42 completed its recovery deployment successfully."
        )

        recalled = await memory_service.recall(
            namespace,
            memory_type="episodic",
            limit=10,
        )

        assert len(recalled) == 1
        created_item = recalled[0]

        assert created_item.content == first_response.output
        assert created_item.namespace == namespace
        assert created_item.memory_type == "episodic"
        assert created_item.metadata == {
            "source": "agent_execution",
            "agent_name": first_agent.definition.name,
            "session_id": "postgres-memory-session-1",
        }

        second_response = await runtime.run(
            second_agent.definition.name,
            AgentRequest(
                input="What happened during the fleet-42 recovery?",
                session_id="postgres-memory-session-2",
                memory_namespace=namespace,
            ),
        )

        assert second_response.output == "Memory retrieved successfully."
        assert second_agent.last_context is not None
        assert second_agent.last_context.memory is not None

        memory = second_agent.last_context.memory

        assert [item.id for item in memory.episodic] == [created_item.id]
        assert memory.episodic[0].content == first_response.output
        assert memory.episodic[0].namespace == namespace

        assert len(second_agent.last_messages) >= 2
        memory_message = second_agent.last_messages[1]

        assert memory_message.role == "system"
        assert first_response.output in memory_message.content

        isolated_response = await runtime.run(
            second_agent.definition.name,
            AgentRequest(
                input="What happened during the fleet-42 recovery?",
                session_id="postgres-memory-session-3",
                memory_namespace="different-postgres-memory-namespace",
            ),
        )

        assert isolated_response.output == "Memory retrieved successfully."
        assert second_agent.last_context is not None
        assert second_agent.last_context.memory is not None
        assert second_agent.last_context.memory.is_empty

    finally:
        if created_item is not None:
            session = session_factory()
            try:
                session.execute(
                    delete(MemoryItemRecord).where(
                        MemoryItemRecord.id == created_item.id,
                    )
                )
                session.commit()
            finally:
                session.close()

        engine.dispose()


asyncio.run(main())
"""

    environment = os.environ.copy()
    environment["POSTGRES_HOST"] = "localhost"
    environment["POSTGRES_PORT"] = "5432"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        "Runtime PostgreSQL memory round-trip integration failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
