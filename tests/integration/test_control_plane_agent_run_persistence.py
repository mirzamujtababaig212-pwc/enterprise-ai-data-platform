"""End-to-end control-plane agent run persistence integration test."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.app import app
from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    AgentRunEventRecord,
    AgentRunRecord,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


API_KEY = os.getenv("API_KEY", "change-me")


def _deterministic_chat_response() -> dict:
    return {
        "reply": "Deterministic integration-test response.",
        "provider": "integration-test",
        "model": "integration-test-model",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "tool_calls": [],
    }


def test_production_control_plane_persists_agent_run_events() -> None:
    """Exercise the real control-plane wiring through PostgreSQL persistence."""

    client = TestClient(app)
    session_id = "production-wiring-integration-session"

    run_id: str | None = None

    try:
        with patch(
            "app.control_plane.dependencies._llm_router.route_chat",
            new=AsyncMock(return_value=_deterministic_chat_response()),
        ) as mock_route_chat:
            response = client.post(
                "/api/v1/agents/enterprise-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                },
                json={
                    "input": "Verify production agent execution persistence.",
                    "session_id": session_id,
                    "user_id": "integration-test-user",
                    "metadata": {
                        "test": "production-control-plane-agent-run",
                    },
                },
            )

        assert response.status_code == 200, response.text
        assert mock_route_chat.await_count == 1

        payload = response.json()

        assert payload["run_id"]
        assert payload["agent_name"] == "enterprise-analyst"
        assert payload["output"] == "Deterministic integration-test response."
        assert payload["session_id"] == session_id

        run_id = payload["run_id"]

        events_response = client.get(
            f"/api/v1/agents/runs/{run_id}/events",
            headers={
                "x-api-key": API_KEY,
            },
        )

        assert events_response.status_code == 200, events_response.text

        events_payload = events_response.json()

        assert [event["event_type"] for event in events_payload["events"]] == [
            "agent.started",
            "llm.requested",
            "llm.completed",
            "agent.completed",
        ]

        assert all(event["run_id"] == run_id for event in events_payload["events"])
        assert all(
            event["agent_name"] == "enterprise-analyst" for event in events_payload["events"]
        )
        assert all(event["session_id"] == session_id for event in events_payload["events"])

        limited_response = client.get(
            f"/api/v1/agents/runs/{run_id}/events",
            params={"limit": 2},
            headers={
                "x-api-key": API_KEY,
            },
        )

        assert limited_response.status_code == 200, limited_response.text
        assert len(limited_response.json()["events"]) == 2

        with SessionLocal() as session:
            events = list(
                session.scalars(
                    select(AgentRunEventRecord)
                    .where(AgentRunEventRecord.run_id == run_id)
                    .order_by(
                        AgentRunEventRecord.created_at.asc(),
                        AgentRunEventRecord.id.asc(),
                    )
                )
            )

        assert [event.event_type for event in events] == [
            "agent.started",
            "llm.requested",
            "llm.completed",
            "agent.completed",
        ]

        assert events
        assert {event.run_id for event in events} == {run_id}
        assert {event.agent_name for event in events} == {"enterprise-analyst"}
        assert {event.session_id for event in events} == {session_id}

        llm_completed = next(event for event in events if event.event_type == "llm.completed")

        assert llm_completed.provider == "integration-test"
        assert llm_completed.model == "integration-test-model"
        assert llm_completed.event_metadata == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

        agent_started = events[0]
        agent_completed = events[-1]

        assert agent_started.provider is None
        assert agent_started.model is None

        assert agent_completed.provider == "integration-test"
        assert agent_completed.model == "integration-test-model"

        with SessionLocal() as session:
            repository = PostgreSQLAgentRunRepository(session)
            run = repository.get(run_id)

        assert run is not None
        assert run.run_id == run_id
        assert run.agent_name == "enterprise-analyst"
        assert run.status is AgentRunStatus.COMPLETED
        assert run.session_id == session_id
        assert run.user_id == "integration-test-user"
        assert run.output == "Deterministic integration-test response."
        assert run.metadata == {
            "test": "production-control-plane-agent-run",
        }

    finally:
        if run_id is not None:
            with SessionLocal() as session:
                session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete()
                session.commit()

            with SessionLocal() as session:
                remaining_events = list(
                    session.scalars(
                        select(AgentRunEventRecord).where(AgentRunEventRecord.run_id == run_id)
                    )
                )

            assert remaining_events == []


def test_production_control_plane_enforces_http_idempotency() -> None:
    """Exercise HTTP idempotency through the real PostgreSQL control plane."""

    client = TestClient(app)

    user_id = "integration-idempotency-user"
    other_user_id = "integration-idempotency-other-user"
    session_id = "production-idempotency-integration-session"
    other_session_id = "production-idempotency-other-session"
    idempotency_key = "production-idempotency-key"

    run_ids: set[str] = set()

    request_payload = {
        "input": "Verify production idempotency behavior.",
        "session_id": session_id,
        "user_id": user_id,
        "metadata": {
            "test": "production-control-plane-idempotency",
        },
    }

    try:
        with patch(
            "app.control_plane.dependencies._llm_router.route_chat",
            new=AsyncMock(return_value=_deterministic_chat_response()),
        ) as mock_route_chat:
            first_response = client.post(
                "/api/v1/agents/enterprise-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                    "Idempotency-Key": idempotency_key,
                },
                json=request_payload,
            )

            assert first_response.status_code == 200, first_response.text

            first_payload = first_response.json()
            first_run_id = first_payload["run_id"]
            run_ids.add(first_run_id)

            assert first_payload["agent_name"] == "enterprise-analyst"
            assert first_payload["output"] == ("Deterministic integration-test response.")
            assert first_payload["session_id"] == session_id
            assert mock_route_chat.await_count == 1

            replay_response = client.post(
                "/api/v1/agents/enterprise-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                    "Idempotency-Key": idempotency_key,
                },
                json=request_payload,
            )

            assert replay_response.status_code == 200, replay_response.text

            replay_payload = replay_response.json()

            assert replay_payload["run_id"] == first_run_id
            assert replay_payload["agent_name"] == "enterprise-analyst"
            assert replay_payload["output"] == ("Deterministic integration-test response.")
            assert replay_payload["session_id"] == session_id
            assert mock_route_chat.await_count == 1

            conflicting_input_response = client.post(
                "/api/v1/agents/enterprise-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                    "Idempotency-Key": idempotency_key,
                },
                json={
                    **request_payload,
                    "input": "A different request must conflict.",
                },
            )

            assert conflicting_input_response.status_code == 409
            assert conflicting_input_response.json() == {
                "detail": "Idempotency key is already associated with a different request."
            }
            assert mock_route_chat.await_count == 1

            conflicting_session_response = client.post(
                "/api/v1/agents/enterprise-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                    "Idempotency-Key": idempotency_key,
                },
                json={
                    **request_payload,
                    "session_id": other_session_id,
                },
            )

            assert conflicting_session_response.status_code == 409
            assert conflicting_session_response.json() == {
                "detail": "Idempotency key is already associated with a different request."
            }
            assert mock_route_chat.await_count == 1

            different_user_response = client.post(
                "/api/v1/agents/enterprise-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                    "Idempotency-Key": idempotency_key,
                },
                json={
                    **request_payload,
                    "user_id": other_user_id,
                },
            )

            assert different_user_response.status_code == 200, different_user_response.text

            different_user_payload = different_user_response.json()
            second_run_id = different_user_payload["run_id"]
            run_ids.add(second_run_id)

            assert second_run_id != first_run_id
            assert different_user_payload["agent_name"] == "enterprise-analyst"
            assert different_user_payload["output"] == ("Deterministic integration-test response.")
            assert different_user_payload["session_id"] == session_id
            assert mock_route_chat.await_count == 2

        with SessionLocal() as session:
            persisted_runs = list(
                session.scalars(
                    select(AgentRunRecord)
                    .where(AgentRunRecord.run_id.in_(run_ids))
                    .order_by(AgentRunRecord.run_id.asc())
                )
            )

        assert len(persisted_runs) == 2

        persisted_by_user = {run.user_id: run for run in persisted_runs}

        first_run = persisted_by_user[user_id]
        second_run = persisted_by_user[other_user_id]

        assert first_run.run_id == first_run_id
        assert first_run.user_id == user_id
        assert first_run.idempotency_key == idempotency_key
        assert first_run.status == AgentRunStatus.COMPLETED
        assert first_run.output == "Deterministic integration-test response."

        assert second_run.run_id == second_run_id
        assert second_run.user_id == other_user_id
        assert second_run.idempotency_key == idempotency_key
        assert second_run.status == AgentRunStatus.COMPLETED
        assert second_run.output == "Deterministic integration-test response."

        with SessionLocal() as session:
            repository = PostgreSQLAgentRunRepository(session)

            assert (
                repository.get_by_idempotency_key(
                    user_id,
                    idempotency_key,
                ).run_id
                == first_run_id
            )

            assert (
                repository.get_by_idempotency_key(
                    other_user_id,
                    idempotency_key,
                ).run_id
                == second_run_id
            )

    finally:
        if run_ids:
            with SessionLocal() as session:
                session.query(AgentRunEventRecord).filter(
                    AgentRunEventRecord.run_id.in_(run_ids)
                ).delete(synchronize_session=False)

                session.query(AgentRunRecord).filter(AgentRunRecord.run_id.in_(run_ids)).delete(
                    synchronize_session=False
                )

                session.commit()

            with SessionLocal() as session:
                remaining_runs = list(
                    session.scalars(
                        select(AgentRunRecord).where(AgentRunRecord.run_id.in_(run_ids))
                    )
                )
                remaining_events = list(
                    session.scalars(
                        select(AgentRunEventRecord).where(AgentRunEventRecord.run_id.in_(run_ids))
                    )
                )

            assert remaining_runs == []
            assert remaining_events == []


def test_production_control_plane_persists_failed_agent_run() -> None:
    """Exercise the real control-plane failure path through PostgreSQL."""

    client = TestClient(app)
    session_id = "production-failure-wiring-integration-session"

    run_id: str | None = None

    async def _failing_route_chat(request: dict) -> dict:
        raise RuntimeError("deterministic provider failure")

    try:
        with patch(
            "app.control_plane.dependencies._llm_router.route_chat",
            new=AsyncMock(side_effect=_failing_route_chat),
        ) as mock_route_chat:
            response = client.post(
                "/api/v1/agents/enterprise-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                },
                json={
                    "input": "Verify production agent failure persistence.",
                    "session_id": session_id,
                    "user_id": "integration-test-user",
                    "metadata": {
                        "test": "production-control-plane-agent-run-failure",
                    },
                },
            )

        assert response.status_code == 409
        assert response.json() == {
            "detail": "deterministic provider failure",
        }
        assert mock_route_chat.await_count == 1

        # The HTTP response does not expose run_id on failure, so locate the
        # just-created run through the persisted integration-test metadata.
        with SessionLocal() as session:
            run_record = session.scalar(
                select(AgentRunRecord)
                .where(
                    AgentRunRecord.agent_name == "enterprise-analyst",
                    AgentRunRecord.session_id == session_id,
                )
                .order_by(AgentRunRecord.started_at.desc())
            )

        assert run_record is not None
        run_id = run_record.run_id

        assert run_record.status == AgentRunStatus.FAILED
        assert run_record.agent_name == "enterprise-analyst"
        assert run_record.session_id == session_id
        assert run_record.user_id == "integration-test-user"
        assert run_record.error_type == "RuntimeError"
        assert run_record.error_message == "deterministic provider failure"
        assert run_record.output is None
        assert run_record.run_metadata == {
            "test": "production-control-plane-agent-run-failure",
        }

        with SessionLocal() as session:
            events = list(
                session.scalars(
                    select(AgentRunEventRecord)
                    .where(AgentRunEventRecord.run_id == run_id)
                    .order_by(
                        AgentRunEventRecord.created_at.asc(),
                        AgentRunEventRecord.id.asc(),
                    )
                )
            )

        assert [event.event_type for event in events] == [
            "agent.started",
            "llm.requested",
            "agent.failed",
        ]

        assert {event.run_id for event in events} == {run_id}
        assert {event.agent_name for event in events} == {"enterprise-analyst"}
        assert {event.session_id for event in events} == {session_id}

        failed_event = events[-1]

        assert failed_event.event_type == "agent.failed"
        assert failed_event.provider is None
        assert failed_event.model is None
        assert failed_event.event_metadata == {
            "error_type": "RuntimeError",
        }

        assert not any(event.event_type == "llm.completed" for event in events)

    finally:
        if run_id is not None:
            with SessionLocal() as session:
                session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete()
                session.commit()

            with SessionLocal() as session:
                remaining_events = list(
                    session.scalars(
                        select(AgentRunEventRecord).where(AgentRunEventRecord.run_id == run_id)
                    )
                )

            assert remaining_events == []


class DeterministicRAGRetriever:
    """Deterministic retriever for production control-plane integration tests."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: dict | None = None,
        governance_policy=None,
    ):
        from rag.models import DocumentChunk, RetrievalResult

        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "min_score": min_score,
                "metadata_filter": metadata_filter,
                "governance_policy": governance_policy,
            }
        )

        return [
            RetrievalResult(
                chunk=DocumentChunk(
                    id="integration-rag-chunk-1",
                    document_id="integration-rag-document-1",
                    content="DELDAI provides an enterprise AI operating layer.",
                    metadata={
                        "source": "integration-test",
                        "document_type": "architecture",
                    },
                    chunk_index=0,
                ),
                score=0.97,
            ),
            RetrievalResult(
                chunk=DocumentChunk(
                    id="integration-rag-chunk-2",
                    document_id="integration-rag-document-1",
                    content="The platform governs agents, models, tools, and enterprise workflows.",
                    metadata={
                        "source": "integration-test",
                        "document_type": "architecture",
                    },
                    chunk_index=1,
                ),
                score=0.91,
            ),
        ]


def test_production_rag_agent_persists_post_tool_checkpoint() -> None:
    """Exercise real RAG tool execution and durable checkpoint persistence."""

    from ai_platform.agents.checkpoint import (
        AgentCheckpointPosition,
        AgentExecutionCheckpoint,
    )
    from ai_platform.agents.tool_calls import AgentToolCall
    from app.control_plane.persistence.models import AgentRunCheckpointRecord

    import app.control_plane.dependencies as dependencies

    client = TestClient(app)
    session_id = "production-rag-checkpoint-integration-session"

    run_id: str | None = None
    original_retriever = dependencies._rag_retriever

    rag_tool = None

    try:
        # The production RAGSearchTool is created during _initialize_agents().
        # Ensure initialization has happened before replacing its retriever.
        import asyncio

        asyncio.run(dependencies._initialize_agents())
        asyncio.run(
            dependencies._tool_authorizer.allow(
                "integration-test-user",
                "rag.search",
            )
        )

        rag_tool = asyncio.run(dependencies._tool_registry.get("rag.search"))

        assert rag_tool is not None
        assert rag_tool.definition.name == "rag.search"

        deterministic_retriever = DeterministicRAGRetriever()

        # RAGSearchTool intentionally keeps its retriever private. This test
        # replaces only that dependency while preserving the real tool and
        # all production control-plane execution layers.
        rag_tool._retriever = deterministic_retriever

        responses = [
            {
                "reply": "",
                "provider": "integration-test",
                "model": "integration-test-model",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 6,
                    "total_tokens": 18,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="call-rag-checkpoint-1",
                        name="rag.search",
                        arguments={
                            "query": "What is DELDAI's enterprise AI architecture?",
                            "top_k": 2,
                        },
                    )
                ],
            },
            {
                "reply": (
                    "The retrieved evidence describes DELDAI as an "
                    "enterprise AI operating layer governing agents, "
                    "models, tools, and workflows."
                ),
                "provider": "integration-test",
                "model": "integration-test-model",
                "usage": {
                    "prompt_tokens": 28,
                    "completion_tokens": 18,
                    "total_tokens": 46,
                },
                "tool_calls": [],
            },
            {
                "reply": (
                    "DELDAI provides an enterprise AI operating layer "
                    "for governing agents, models, tools, and workflows."
                ),
                "provider": "integration-test",
                "model": "integration-test-model",
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 20,
                    "total_tokens": 60,
                },
                "tool_calls": [],
            },
        ]

        async def _deterministic_route_chat(request: dict) -> dict:
            return responses.pop(0)

        with patch(
            "app.control_plane.dependencies._llm_router.route_chat",
            new=AsyncMock(side_effect=_deterministic_route_chat),
        ) as mock_route_chat:
            response = client.post(
                "/api/v1/agents/enterprise-rag-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                },
                json={
                    "input": "Explain DELDAI's enterprise AI architecture.",
                    "session_id": session_id,
                    "user_id": "integration-test-user",
                    "metadata": {
                        "test": "production-rag-checkpoint-persistence",
                    },
                },
            )

        assert response.status_code == 200, response.text
        assert mock_route_chat.await_count == 3
        assert responses == []

        payload = response.json()

        assert payload["run_id"]
        assert payload["agent_name"] == "enterprise-rag-analyst"
        assert payload["session_id"] == session_id
        assert (
            payload["output"] == "DELDAI provides an enterprise AI operating layer "
            "for governing agents, models, tools, and workflows."
        )

        run_id = payload["run_id"]

        assert len(deterministic_retriever.calls) == 1
        assert deterministic_retriever.calls[0]["query"] == (
            "What is DELDAI's enterprise AI architecture?"
        )
        assert deterministic_retriever.calls[0]["top_k"] == 2
        assert deterministic_retriever.calls[0]["min_score"] is None
        assert deterministic_retriever.calls[0]["metadata_filter"] is None

        with SessionLocal() as session:
            run_record = session.scalar(
                select(AgentRunRecord).where(
                    AgentRunRecord.run_id == run_id,
                )
            )

            events = list(
                session.scalars(
                    select(AgentRunEventRecord)
                    .where(AgentRunEventRecord.run_id == run_id)
                    .order_by(
                        AgentRunEventRecord.created_at.asc(),
                        AgentRunEventRecord.id.asc(),
                    )
                )
            )

            checkpoints = list(
                session.scalars(
                    select(AgentRunCheckpointRecord)
                    .where(AgentRunCheckpointRecord.run_id == run_id)
                    .order_by(
                        AgentRunCheckpointRecord.created_at.asc(),
                        AgentRunCheckpointRecord.id.asc(),
                    )
                )
            )

        assert run_record is not None
        assert run_record.status == AgentRunStatus.COMPLETED
        assert run_record.agent_name == "enterprise-rag-analyst"
        assert run_record.session_id == session_id
        assert run_record.user_id == "integration-test-user"

        assert [event.event_type for event in events] == [
            "agent.started",
            "llm.requested",
            "llm.completed",
            "tool.call.requested",
            "tool.authorization.decision",
            "tool.call.completed",
            "llm.requested",
            "llm.completed",
            "agent.completed",
            "llm.requested",
            "llm.completed",
            "agent.completed",
        ]

        assert all(event.run_id == run_id for event in events)
        assert all(event.agent_name == "enterprise-rag-analyst" for event in events)
        assert all(event.session_id == session_id for event in events)

        tool_requested = next(
            event for event in events if event.event_type == "tool.call.requested"
        )
        tool_completed = next(
            event for event in events if event.event_type == "tool.call.completed"
        )

        assert tool_requested.tool_name == "rag.search"
        assert tool_requested.call_id == "call-rag-checkpoint-1"
        assert tool_requested.tool_round == 1

        assert tool_completed.tool_name == "rag.search"
        assert tool_completed.call_id == "call-rag-checkpoint-1"
        assert tool_completed.tool_round == 1

        assert len(checkpoints) == 2
        assert [checkpoint.position for checkpoint in checkpoints] == [
            AgentCheckpointPosition.BEFORE_TOOL_EXECUTION.value,
            AgentCheckpointPosition.AFTER_TOOL_EXECUTION.value,
        ]
        assert [checkpoint.tool_round for checkpoint in checkpoints] == [1, 1]

        before_checkpoint_record = checkpoints[0]

        assert before_checkpoint_record.run_id == run_id
        assert before_checkpoint_record.agent_name == "enterprise-rag-analyst"
        assert before_checkpoint_record.session_id == session_id
        assert before_checkpoint_record.user_id == "integration-test-user"
        assert (
            before_checkpoint_record.schema_version
            == AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION
        )

        before_checkpoint = AgentExecutionCheckpoint.from_dict(
            dict(before_checkpoint_record.checkpoint_payload)
        )

        assert before_checkpoint.run_id == run_id
        assert before_checkpoint.agent_name == "enterprise-rag-analyst"
        assert before_checkpoint.session_id == session_id
        assert before_checkpoint.user_id == "integration-test-user"
        assert before_checkpoint.schema_version == AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION
        assert before_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
        assert before_checkpoint.tool_round == 1
        assert before_checkpoint.messages

        after_checkpoint_record = checkpoints[1]

        assert after_checkpoint_record.run_id == run_id
        assert after_checkpoint_record.agent_name == "enterprise-rag-analyst"
        assert after_checkpoint_record.session_id == session_id
        assert after_checkpoint_record.user_id == "integration-test-user"
        assert (
            after_checkpoint_record.schema_version
            == AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION
        )
        assert (
            after_checkpoint_record.position == AgentCheckpointPosition.AFTER_TOOL_EXECUTION.value
        )
        assert after_checkpoint_record.tool_round == 1

        checkpoint = AgentExecutionCheckpoint.from_dict(
            dict(after_checkpoint_record.checkpoint_payload)
        )

        assert checkpoint.run_id == run_id
        assert checkpoint.agent_name == "enterprise-rag-analyst"
        assert checkpoint.session_id == session_id
        assert checkpoint.user_id == "integration-test-user"
        assert checkpoint.schema_version == AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION
        assert checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
        assert checkpoint.tool_round == 1

        assert checkpoint.messages
        tool_message = checkpoint.messages[-1]

        assert tool_message.role.value == "tool"

        import json

        tool_payload = json.loads(tool_message.content)

        assert tool_payload["call_id"] == "call-rag-checkpoint-1"
        assert tool_payload["tool_name"] == "rag.search"
        assert tool_payload["success"] is True

        assert tool_payload["output"]["query"] == ("What is DELDAI's enterprise AI architecture?")
        assert tool_payload["output"]["retrieved_count"] == 2

        retrieved_results = tool_payload["output"]["results"]

        assert len(retrieved_results) == 2
        assert retrieved_results[0]["chunk_id"] == "integration-rag-chunk-1"
        assert retrieved_results[0]["document_id"] == "integration-rag-document-1"
        assert retrieved_results[0]["content"] == (
            "DELDAI provides an enterprise AI operating layer."
        )
        assert retrieved_results[0]["score"] == 0.97

        assert retrieved_results[1]["chunk_id"] == "integration-rag-chunk-2"
        assert retrieved_results[1]["document_id"] == "integration-rag-document-1"
        assert retrieved_results[1]["content"] == (
            "The platform governs agents, models, tools, and enterprise workflows."
        )
        assert retrieved_results[1]["score"] == 0.91

        # Verify the checkpoint contains the retrieval output but does not
        # contain execution infrastructure objects.
        assert "governance_policy" not in checkpoint.metadata
        assert "request_metadata" not in checkpoint.metadata

        with SessionLocal() as session:
            repository = PostgreSQLAgentRunRepository(session)
            persisted_run = repository.get(run_id)

        assert persisted_run is not None
        assert persisted_run.status is AgentRunStatus.COMPLETED

        # Parent deletion must cascade to durable events and checkpoints.
        with SessionLocal() as session:
            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete()
            session.commit()

        with SessionLocal() as session:
            remaining_events = list(
                session.scalars(
                    select(AgentRunEventRecord).where(AgentRunEventRecord.run_id == run_id)
                )
            )
            remaining_checkpoints = list(
                session.scalars(
                    select(AgentRunCheckpointRecord).where(
                        AgentRunCheckpointRecord.run_id == run_id
                    )
                )
            )

        assert remaining_events == []
        assert remaining_checkpoints == []

        run_id = None

    finally:
        asyncio.run(
            dependencies._tool_authorizer.deny(
                "integration-test-user",
                "rag.search",
            )
        )

        if rag_tool is not None:
            rag_tool._retriever = original_retriever

        if run_id is not None:
            with SessionLocal() as session:
                session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete()
                session.commit()


def test_production_rag_checkpoint_failure_preserves_postgres_tool_result() -> None:
    """Verify checkpoint failure does not lose a completed tool execution."""

    import asyncio

    from ai_platform.agents.tool_calls import AgentToolCall
    from app.control_plane.agent_runs.postgres_repository import (
        PostgreSQLAgentRunRepository,
    )
    from app.control_plane.persistence.models import (
        AgentRunCheckpointRecord,
        ToolExecutionIdempotencyRecord,
    )
    from app.control_plane.tool_execution.postgres_idempotency import (
        PostgreSQLToolExecutionIdempotencyStore,
    )
    from tools.execution.context import ToolExecutionContext
    from tools.execution.service import ToolExecutionService

    import app.control_plane.dependencies as dependencies

    client = TestClient(app)
    session_id = "production-rag-checkpoint-failure-integration-session"

    run_id: str | None = None
    rag_tool = None
    original_retriever = None

    try:
        asyncio.run(dependencies._initialize_agents())
        asyncio.run(
            dependencies._tool_authorizer.allow(
                "integration-test-user",
                "rag.search",
            )
        )

        rag_tool = asyncio.run(dependencies._tool_registry.get("rag.search"))

        assert rag_tool is not None
        assert rag_tool.definition.name == "rag.search"

        original_retriever = rag_tool._retriever
        deterministic_retriever = DeterministicRAGRetriever()
        rag_tool._retriever = deterministic_retriever

        agent = asyncio.run(dependencies._agent_registry.get("enterprise-rag-analyst"))
        assert agent is not None

        assert agent._checkpoint_handler is dependencies._agent_checkpoint_handler

        responses = [
            {
                "reply": "",
                "provider": "integration-test",
                "model": "integration-test-model",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 6,
                    "total_tokens": 18,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="call-rag-checkpoint-failure-1",
                        name="rag.search",
                        arguments={
                            "query": "What is DELDAI's enterprise AI architecture?",
                            "top_k": 2,
                        },
                    )
                ],
            },
            {
                "reply": (
                    "DELDAI provides an enterprise AI operating layer "
                    "for governing agents, models, tools, and workflows."
                ),
                "provider": "integration-test",
                "model": "integration-test-model",
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 20,
                    "total_tokens": 60,
                },
                "tool_calls": [],
            },
        ]

        async def _deterministic_route_chat(request: dict) -> dict:
            return responses.pop(0)

        original_checkpoint_handler_save = dependencies._agent_checkpoint_handler.save
        checkpoint_save_calls = 0

        async def _failing_checkpoint_handler_save(
            checkpoint,
            *,
            lease_id=None,
        ):
            nonlocal checkpoint_save_calls
            checkpoint_save_calls += 1

            if checkpoint_save_calls == 1:
                return await original_checkpoint_handler_save(
                    checkpoint,
                    lease_id=lease_id,
                )

            raise RuntimeError("deterministic checkpoint persistence failure")

        with (
            patch(
                "app.control_plane.dependencies._llm_router.route_chat",
                new=AsyncMock(side_effect=_deterministic_route_chat),
            ) as mock_route_chat,
            patch.object(
                dependencies._agent_checkpoint_handler,
                "save",
                new=_failing_checkpoint_handler_save,
            ),
        ):
            response = client.post(
                "/api/v1/agents/enterprise-rag-analyst/run",
                headers={
                    "x-api-key": API_KEY,
                },
                json={
                    "input": "Explain DELDAI's enterprise AI architecture.",
                    "session_id": session_id,
                    "user_id": "integration-test-user",
                    "metadata": {
                        "test": "production-rag-checkpoint-failure",
                    },
                },
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "deterministic checkpoint persistence failure"
        assert mock_route_chat.await_count == 1
        assert len(responses) == 1

        assert len(deterministic_retriever.calls) == 1

        with SessionLocal() as session:
            run_record = session.scalar(
                select(AgentRunRecord).where(
                    AgentRunRecord.session_id == session_id,
                    AgentRunRecord.user_id == "integration-test-user",
                )
            )

            assert run_record is not None
            run_id = run_record.run_id

            run_repository = PostgreSQLAgentRunRepository(session)
            run = run_repository.get(run_id)

            checkpoints = list(
                session.scalars(
                    select(AgentRunCheckpointRecord).where(
                        AgentRunCheckpointRecord.run_id == run_id
                    )
                )
            )

            idempotency_records = list(
                session.scalars(
                    select(ToolExecutionIdempotencyRecord).where(
                        ToolExecutionIdempotencyRecord.run_id == run_id,
                        ToolExecutionIdempotencyRecord.call_id == "call-rag-checkpoint-failure-1",
                        ToolExecutionIdempotencyRecord.tool_name == "rag.search",
                    )
                )
            )

        assert run is not None
        assert run.status is AgentRunStatus.FAILED
        assert run.error_type == "RuntimeError", (
            f"error_type={run.error_type!r}, " f"error_message={run.error_message!r}"
        )
        assert run.error_message == "deterministic checkpoint persistence failure"

        # The first checkpoint was persisted before the tool execution.
        # The second checkpoint failed after the tool result was durably stored.
        assert len(checkpoints) == 1
        assert checkpoints[0].position == "before_tool_execution"

        # The tool result must nevertheless be durable.
        assert len(idempotency_records) == 1

        idempotency_record = idempotency_records[0]

        assert idempotency_record.status == "completed"
        assert idempotency_record.success is True
        assert idempotency_record.output["query"] == (
            "What is DELDAI's enterprise AI architecture?"
        )
        assert idempotency_record.output["retrieved_count"] == 2

        # Reconstruct the production PostgreSQL-backed execution service and
        # replay the same logical tool call. The durable result must be
        # returned without invoking the tool again.
        replay_store = PostgreSQLToolExecutionIdempotencyStore(SessionLocal)
        replay_service = ToolExecutionService(
            dependencies._tool_registry,
            idempotency_store=replay_store,
        )

        replay_context = ToolExecutionContext(
            run_id=run_id,
            call_id="call-rag-checkpoint-failure-1",
            agent_name="enterprise-rag-analyst",
            session_id=session_id,
            user_id="integration-test-user",
            request_metadata={
                "test": "production-rag-checkpoint-failure",
            },
        )

        replay = asyncio.run(
            replay_service.execute(
                "rag.search",
                {
                    "query": "What is DELDAI's enterprise AI architecture?",
                    "top_k": 2,
                },
                principal="integration-test-user",
                execution_context=replay_context,
            )
        )

        assert replay.success is True
        assert replay.output == idempotency_record.output

        # The original tool execution happened exactly once. The replay was
        # served entirely from PostgreSQL idempotency state despite the
        # post-tool checkpoint persistence failure.
        assert len(deterministic_retriever.calls) == 1

    finally:
        if rag_tool is not None and original_retriever is not None:
            rag_tool._retriever = original_retriever

        if run_id is not None:
            with SessionLocal() as session:
                session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete()
                session.commit()


def test_production_agent_run_recovers_from_postgres_checkpoint() -> None:
    """Recover a failed agent run from its real PostgreSQL checkpoint."""

    import asyncio

    from ai_platform.agents.checkpoint import (
        AgentCheckpointPosition,
        AgentExecutionCheckpoint,
    )
    from ai_platform.agents.llm_messages import AgentMessage, AgentMessageRole
    from app.control_plane.agent_runs.recovery_service import (
        AgentRunRecoveryService,
    )
    from app.control_plane.agent_runs.request_snapshot import (
        AgentRunRequestSnapshot,
    )
    from app.control_plane.agent_runs.postgres_repository import (
        PostgreSQLAgentRunRepository,
    )
    from app.control_plane.agent_checkpoints.postgres_repository import (
        PostgreSQLAgentCheckpointsRepository,
    )
    from app.control_plane.persistence.models import (
        AgentRunCheckpointRecord,
    )

    import app.control_plane.dependencies as dependencies

    run_id = "integration-recovery-postgres-run"
    session_id = "integration-recovery-session"
    recovery_run_session = None
    recovery_checkpoint_session = None

    try:
        asyncio.run(dependencies._initialize_agents())

        request_payload = {
            "input": "Continue the vehicle risk analysis.",
            "session_id": session_id,
            "user_id": "integration-test-user",
            "metadata": {
                "test": "production-agent-run-recovery",
            },
        }

        from ai_platform.agents.models import AgentRequest

        request = AgentRequest(
            input=request_payload["input"],
            session_id=request_payload["session_id"],
            user_id=request_payload["user_id"],
            metadata=request_payload["metadata"],
        )

        snapshot = AgentRunRequestSnapshot.from_request(request)

        with SessionLocal() as session:
            run_repository = PostgreSQLAgentRunRepository(session)

            failed_run = AgentRun(
                run_id=run_id,
                agent_name="enterprise-analyst",
                session_id=session_id,
                user_id="integration-test-user",
                status=AgentRunStatus.FAILED,
                error_type="RuntimeError",
                error_message="simulated interrupted execution",
                metadata=request_payload["metadata"],
                request_snapshot=snapshot,
            )

            run_repository.create(failed_run)

        checkpoint = AgentExecutionCheckpoint(
            schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
            run_id=run_id,
            agent_name="enterprise-analyst",
            session_id=session_id,
            user_id="integration-test-user",
            messages=(
                AgentMessage(
                    role=AgentMessageRole.USER,
                    content="Find the vehicle risk.",
                ),
                AgentMessage(
                    role=AgentMessageRole.ASSISTANT,
                    content="I retrieved the relevant vehicle information.",
                ),
                AgentMessage(
                    role=AgentMessageRole.TOOL,
                    content=(
                        '{"call_id":"call-recovery-1",'
                        '"tool_name":"rag.search",'
                        '"success":true,'
                        '"output":{"query":"vehicle risk",'
                        '"retrieved_count":1}}'
                    ),
                ),
            ),
            tool_round=1,
            position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
            metadata={
                "source": "integration-test",
            },
        )

        with SessionLocal() as session:
            checkpoint_repository = PostgreSQLAgentCheckpointsRepository(session)
            checkpoint_repository.save(checkpoint)

        recovery_run_session = SessionLocal()
        recovery_checkpoint_session = SessionLocal()

        recovery_service = AgentRunRecoveryService(
            runtime=dependencies._agent_runtime,
            repository=PostgreSQLAgentRunRepository(
                recovery_run_session,
            ),
            checkpoints_repository=PostgreSQLAgentCheckpointsRepository(
                recovery_checkpoint_session,
            ),
        )
        with patch(
            "app.control_plane.dependencies._llm_router.route_chat",
            new=AsyncMock(
                return_value={
                    "reply": "Recovered successfully from the durable checkpoint.",
                    "provider": "integration-test",
                    "model": "integration-test-model",
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                    "tool_calls": [],
                }
            ),
        ) as mock_route_chat:
            result = asyncio.run(
                recovery_service.recover(run_id),
            )

        assert result.run_id == run_id
        assert result.response.output == ("Recovered successfully from the durable checkpoint.")
        assert mock_route_chat.await_count == 1

        with SessionLocal() as session:
            run_repository = PostgreSQLAgentRunRepository(session)
            restored_run = run_repository.get(run_id)

            checkpoints = list(
                session.scalars(
                    select(AgentRunCheckpointRecord)
                    .where(AgentRunCheckpointRecord.run_id == run_id)
                    .order_by(
                        AgentRunCheckpointRecord.created_at.asc(),
                        AgentRunCheckpointRecord.id.asc(),
                    )
                )
            )

        assert restored_run is not None
        assert restored_run.status is AgentRunStatus.COMPLETED
        assert restored_run.output == ("Recovered successfully from the durable checkpoint.")
        assert restored_run.error_type is None
        assert restored_run.error_message is None
        assert restored_run.completed_at is not None

        assert len(checkpoints) == 1
        assert checkpoints[0].run_id == run_id

        # Recovery must not execute the already-completed tool again.
        assert mock_route_chat.await_count == 1

        # A completed run cannot be claimed for a second recovery.
        with pytest.raises(ValueError, match="not eligible for recovery"):
            asyncio.run(
                recovery_service.recover(run_id),
            )

    finally:
        if recovery_run_session is not None:
            recovery_run_session.close()

        if recovery_checkpoint_session is not None:
            recovery_checkpoint_session.close()

        with SessionLocal() as session:
            session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete()
            session.commit()
