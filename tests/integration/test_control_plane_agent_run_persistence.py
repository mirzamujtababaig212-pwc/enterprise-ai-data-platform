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
    AgentEvaluationRunRecord,
    AgentRunEventRecord,
    AgentRunRecord,
    AgentRunStepRecord,
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


def test_production_control_plane_persists_and_reads_agent_evaluation() -> None:
    """Exercise production agent execution and evaluation through PostgreSQL."""

    client = TestClient(app)
    session_id = "production-evaluation-integration-session"

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
                    "input": "Evaluate production agent execution.",
                    "session_id": session_id,
                    "user_id": "integration-evaluation-user",
                    "metadata": {
                        "test": "production-control-plane-agent-evaluation",
                    },
                },
            )

        assert response.status_code == 200, response.text
        assert mock_route_chat.await_count == 1

        run_payload = response.json()

        assert run_payload["run_id"]
        assert run_payload["agent_name"] == "enterprise-analyst"
        assert run_payload["output"] == "Deterministic integration-test response."
        assert run_payload["session_id"] == session_id

        run_id = run_payload["run_id"]

        evaluation_response = client.post(
            f"/api/v1/agents/runs/{run_id}/evaluations",
            headers={
                "x-api-key": API_KEY,
            },
            json={
                "expected_answer": "Deterministic integration-test response.",
                "max_execution_time_ms": 60_000,
                "max_steps_per_run": 10,
                "max_invalid_tool_calls": 0,
                "allow_governance_denials": False,
                "require_task_completed": True,
                "require_answer_match": True,
                "name": "production-integration-quality-gate",
            },
        )

        assert evaluation_response.status_code == 200, evaluation_response.text

        evaluation_payload = evaluation_response.json()

        assert evaluation_payload["evaluation_run_id"]
        assert evaluation_payload["passed"] is True

        assert evaluation_payload["lineage"] == {
            "evaluated_run_id": run_id,
            "agent_name": "enterprise-analyst",
            "agent_version": None,
            "tenant_id": "tenant-a",
            "effective_model": "gpt-4.1-mini",
            "effective_provider": None,
            "model_policy_id": None,
            "model_policy_version": None,
        }

        metrics = evaluation_payload["metrics"]

        assert metrics["steps_total"] == 0
        assert metrics["tool_calls_total"] == 0
        assert metrics["tool_calls_successful"] == 0
        assert metrics["tool_calls_failed"] == 0
        assert metrics["invalid_tool_calls"] == 0
        assert metrics["governance_denials"] == 0
        assert metrics["task_completed"] is True
        assert metrics["execution_time_ms"] >= 0

        assert evaluation_payload["policy"] == {
            "max_execution_time_ms": 60_000,
            "max_steps_per_run": 10,
            "max_invalid_tool_calls": 0,
            "allow_governance_denials": False,
            "require_task_completed": True,
            "require_answer_match": True,
            "name": "production-integration-quality-gate",
        }

        assert evaluation_payload["quality_gate"] == {
            "passed": True,
            "violations": [],
        }

        assert evaluation_payload["answer_evaluation"] == {
            "evaluated": True,
            "exact_match": True,
            "normalization": "whitespace_casefold",
        }

        evaluation_run_id = evaluation_payload["evaluation_run_id"]

        with SessionLocal() as session:
            evaluation_record = session.get(
                AgentEvaluationRunRecord,
                evaluation_run_id,
            )

            assert evaluation_record is not None
            assert evaluation_record.evaluation_run_id == evaluation_run_id
            assert evaluation_record.evaluated_run_id == run_id
            assert evaluation_record.agent_name == "enterprise-analyst"
            assert evaluation_record.agent_version is None
            assert evaluation_record.tenant_id == "tenant-a"
            assert evaluation_record.passed is True

            assert evaluation_record.lineage == {
                "evaluated_run_id": run_id,
                "agent_name": "enterprise-analyst",
                "agent_version": None,
                "tenant_id": "tenant-a",
                "effective_model": "gpt-4.1-mini",
                "effective_provider": None,
                "model_policy_id": None,
                "model_policy_version": None,
            }

            assert evaluation_record.metrics["task_completed"] is True
            assert evaluation_record.metrics["tool_calls_total"] == 0
            assert evaluation_record.metrics["invalid_tool_calls"] == 0

            assert evaluation_record.policy == {
                "max_execution_time_ms": 60_000,
                "max_steps_per_run": 10,
                "max_invalid_tool_calls": 0,
                "allow_governance_denials": False,
                "require_task_completed": True,
                "require_answer_match": True,
                "name": "production-integration-quality-gate",
            }

            assert evaluation_record.quality_gate == {
                "passed": True,
                "violations": [],
            }

            assert evaluation_record.answer_evaluation == {
                "evaluated": True,
                "exact_match": True,
                "normalization": "whitespace_casefold",
            }

        list_response = client.get(
            f"/api/v1/agents/runs/{run_id}/evaluations",
            params={"limit": 10},
            headers={
                "x-api-key": API_KEY,
            },
        )

        assert list_response.status_code == 200, list_response.text

        list_payload = list_response.json()

        assert list_payload["limit"] == 10
        assert len(list_payload["evaluations"]) == 1

        listed_evaluation = list_payload["evaluations"][0]

        assert listed_evaluation == evaluation_payload

        with SessionLocal() as session:
            run_record = session.get(AgentRunRecord, run_id)

            assert run_record is not None
            assert run_record.status == "completed"
            assert run_record.tenant_id == "tenant-a"

            evaluation_count = (
                session.query(AgentEvaluationRunRecord)
                .filter(AgentEvaluationRunRecord.evaluated_run_id == run_id)
                .count()
            )

            assert evaluation_count == 1

    finally:
        if run_id is not None:
            with SessionLocal() as session:
                session.query(AgentEvaluationRunRecord).filter(
                    AgentEvaluationRunRecord.evaluated_run_id == run_id
                ).delete(synchronize_session=False)

                session.query(AgentRunStepRecord).filter(
                    AgentRunStepRecord.run_id == run_id
                ).delete(synchronize_session=False)

                session.query(AgentRunEventRecord).filter(
                    AgentRunEventRecord.run_id == run_id
                ).delete(synchronize_session=False)

                session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                    synchronize_session=False
                )

                session.commit()


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
            "governance.decision",
            "governance.decision",
            "governance.decision",
            "agent.started",
            "llm.requested",
            "llm.completed",
            "runtime.decision",
            "agent.completed",
        ]

        runtime_decision_payload = next(
            event for event in events_payload["events"] if event["event_type"] == "runtime.decision"
        )

        assert runtime_decision_payload["run_id"] == run_id
        assert runtime_decision_payload["session_id"] == session_id
        assert runtime_decision_payload["user_id"] == "integration-test-user"
        assert runtime_decision_payload["provider"] == "integration-test"
        assert runtime_decision_payload["model"] == "integration-test-model"
        assert runtime_decision_payload["step_index"] == 0
        assert runtime_decision_payload["metadata"] == {
            "decision": "stop",
            "reason": "iteration_budget_exhausted",
            "iteration": 1,
        }

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
            "governance.decision",
            "governance.decision",
            "governance.decision",
            "agent.started",
            "llm.requested",
            "llm.completed",
            "runtime.decision",
            "agent.completed",
        ]

        runtime_decision = next(event for event in events if event.event_type == "runtime.decision")

        assert runtime_decision.run_id == run_id
        assert runtime_decision.session_id == session_id
        assert runtime_decision.user_id == "integration-test-user"
        assert runtime_decision.provider == "integration-test"
        assert runtime_decision.model == "integration-test-model"
        assert runtime_decision.step_index == 0
        assert runtime_decision.event_metadata == {
            "decision": "stop",
            "reason": "iteration_budget_exhausted",
            "iteration": 1,
        }

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
                    "tenant-a",
                    user_id,
                    idempotency_key,
                ).run_id
                == first_run_id
            )

            assert (
                repository.get_by_idempotency_key(
                    "tenant-a",
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
            "governance.decision",
            "governance.decision",
            "governance.decision",
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

        from app.control_plane.auth import principal_from_api_key

        authenticated_principal = principal_from_api_key(API_KEY)

        asyncio.run(
            dependencies._tool_authorizer.allow(
                authenticated_principal,
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

            steps = list(
                session.scalars(
                    select(AgentRunStepRecord)
                    .where(AgentRunStepRecord.run_id == run_id)
                    .order_by(
                        AgentRunStepRecord.step_index.asc(),
                        AgentRunStepRecord.created_at.asc(),
                        AgentRunStepRecord.step_id.asc(),
                    )
                )
            )

        assert run_record is not None
        assert run_record.status == AgentRunStatus.COMPLETED
        assert run_record.agent_name == "enterprise-rag-analyst"
        assert run_record.session_id == session_id
        assert run_record.user_id == "integration-test-user"

        assert len(steps) == 3

        assert [(step.step_id, step.step_index, step.status) for step in steps] == [
            ("retrieve_evidence", 0, "completed"),
            ("analyze_evidence", 1, "completed"),
            ("produce_answer", 2, "completed"),
        ]

        assert all(step.run_id == run_id for step in steps)
        # Agent identity is asserted on the parent AgentRunRecord above.
        assert all(step.attempt == 1 for step in steps)
        assert all(step.completed_at is not None for step in steps)

        retrieve_step = steps[0]

        assert retrieve_step.step_type == "tool"
        assert retrieve_step.tool_name == "rag.search"
        assert retrieve_step.call_id == "call-rag-checkpoint-1"
        assert retrieve_step.status == "completed"
        assert retrieve_step.output is not None

        analyze_step = steps[1]

        assert analyze_step.step_type == "model"
        assert analyze_step.tool_name is None
        assert analyze_step.call_id is None
        assert analyze_step.status == "completed"

        produce_step = steps[2]

        assert produce_step.step_type == "model"
        assert produce_step.tool_name is None
        assert produce_step.call_id is None
        assert produce_step.status == "completed"

        assert [event.event_type for event in events] == [
            "governance.decision",
            "governance.decision",
            "governance.decision",
            "agent.started",
            "orchestration.step.started",
            "llm.requested",
            "llm.completed",
            "tool.call.requested",
            "tool.authorization.decision",
            "governance.decision",
            "governance.decision",
            "tool.call.completed",
            "orchestration.step.completed",
            "orchestration.step.started",
            "llm.requested",
            "llm.completed",
            "orchestration.step.completed",
            "orchestration.step.started",
            "llm.requested",
            "llm.completed",
            "orchestration.step.completed",
            "runtime.decision",
            "agent.completed",
        ]

        orchestration_events = [
            event for event in events if event.event_type.startswith("orchestration.step.")
        ]

        assert [event.event_type for event in orchestration_events] == [
            "orchestration.step.started",
            "orchestration.step.completed",
            "orchestration.step.started",
            "orchestration.step.completed",
            "orchestration.step.started",
            "orchestration.step.completed",
        ]

        assert [
            (event.step_id, event.step_index, event.step_name) for event in orchestration_events
        ] == [
            ("retrieve_evidence", 0, "Retrieve enterprise evidence"),
            ("retrieve_evidence", 0, "Retrieve enterprise evidence"),
            ("analyze_evidence", 1, "Analyze retrieved evidence"),
            ("analyze_evidence", 1, "Analyze retrieved evidence"),
            ("produce_answer", 2, "Produce grounded answer"),
            ("produce_answer", 2, "Produce grounded answer"),
        ]

        assert all(event.run_id == run_id for event in events)
        assert all(event.session_id == session_id for event in events)

        execution_events = [event for event in events if event.event_type != "governance.decision"]
        assert all(event.agent_name == "enterprise-rag-analyst" for event in execution_events)
        assert sum(event.event_type == "governance.decision" for event in events) == 5

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

        from app.control_plane.auth import principal_from_api_key

        authenticated_principal = principal_from_api_key(API_KEY)

        asyncio.run(
            dependencies._tool_authorizer.allow(
                authenticated_principal,
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


@pytest.mark.asyncio
async def test_production_mcp_agent_run_recovers_from_postgres_checkpoint() -> None:
    """Exercise durable MCP execution recovery through the production stack."""

    import json
    import sys
    from pathlib import Path

    import httpx

    from ai_platform.agents.checkpoint import AgentCheckpointPosition
    from ai_platform.agents.llm_agent import LLMAgent
    from ai_platform.agents.models import AgentDefinition
    from ai_platform.agents.tool_calls import AgentToolCall
    from ai_platform.llm_gateway.config.settings import settings
    from app.control_plane.auth import principal_from_api_key
    from app.control_plane.persistence.models import (
        AgentRunCheckpointRecord,
        ToolExecutionIdempotencyRecord,
    )
    import app.control_plane.dependencies as dependencies
    from tools.mcp.sdk_client import MCPPythonSDKClient

    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    )

    agent_name = "enterprise-mcp-recovery-test-agent"
    session_id = "production-mcp-recovery-integration-session"
    call_id = "call-mcp-recovery-1"

    run_id: str | None = None

    original_checkpoint_save = dependencies._agent_checkpoint_handler.save
    original_idempotency_complete = dependencies._tool_idempotency_store.complete
    original_idempotency_release = dependencies._tool_idempotency_store.release
    original_mcp_call_tool = MCPPythonSDKClient.call_tool
    original_mcp_servers = os.environ.get("MCP_SERVERS")

    checkpoint_save_count = 0
    mcp_invocations: list[dict] = []

    authenticated_principal = principal_from_api_key(API_KEY)

    async def _failing_checkpoint_save(checkpoint, *, lease_id=None):
        nonlocal checkpoint_save_count

        checkpoint_save_count += 1

        # BEFORE_TOOL_EXECUTION must survive so recovery has a durable
        # replay point. Fail only when the runtime attempts to persist
        # AFTER_TOOL_EXECUTION.
        if checkpoint.position == AgentCheckpointPosition.AFTER_TOOL_EXECUTION:
            raise RuntimeError("deterministic MCP checkpoint persistence failure")

        return await original_checkpoint_save(
            checkpoint,
            lease_id=lease_id,
        )

    async def _failing_idempotency_complete(
        key,
        result,
        *,
        claim_token=None,
    ):
        # The real MCP call has already completed at this point. Failing
        # completion deliberately leaves the durable claim unresolved.
        raise RuntimeError("deterministic MCP idempotency completion failure")

    async def _preserve_claim_on_release(
        key,
        *,
        claim_token=None,
    ):
        # Model the crash window: the execution claim cannot be released
        # after the external MCP call has already happened.
        return None

    async def _counting_mcp_call_tool(
        self,
        name,
        arguments,
        *,
        meta=None,
    ):
        mcp_invocations.append(
            {
                "name": name,
                "arguments": dict(arguments),
                "meta": meta,
            }
        )
        return await original_mcp_call_tool(
            self,
            name,
            arguments,
            meta=meta,
        )

    try:
        # Ensure the normal production agents exist first.
        await dependencies._initialize_agents()

        # The repository .env intentionally has no MCP_SERVERS configured.
        # For this integration test, provide the real MCP fixture through
        # the same production MCP_SERVERS configuration path.
        search_server = (
            Path(__file__).resolve().parents[1] / "tools" / "mcp" / "fixtures" / "test_server.py"
        )

        assert search_server.exists(), search_server

        mcp_servers_payload = [
            {
                "name": "document-server",
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(search_server)],
            }
        ]

        # Reset any MCP initialization that may have happened earlier
        # with the repository's normal empty MCP configuration.
        await dependencies.close_mcp_servers()

        os.environ["MCP_SERVERS"] = json.dumps(mcp_servers_payload)

        # This is the real production initialization/discovery path.
        await dependencies.initialize_mcp_servers()

        mcp_tool = await dependencies._tool_registry.get("search_documents")

        assert mcp_tool is not None
        assert mcp_tool.definition.name == "search_documents"

        # Use a dedicated test agent so production agent capabilities are
        # not changed merely to exercise this integration path.
        test_definition = AgentDefinition(
            name=agent_name,
            description="Production MCP durable-recovery integration test agent.",
            system_prompt=(
                "You are an enterprise document analysis agent. "
                "Use search_documents when document evidence is requested. "
                "After receiving tool results, provide a concise final answer."
            ),
            model=settings.DEFAULT_CHAT_MODEL,
            temperature=0.0,
            max_tokens=1024,
            tool_names=("search_documents",),
        )

        awaitable_agent = LLMAgent(
            test_definition,
            observer=dependencies._agent_observer,
            checkpoint_handler=dependencies._agent_checkpoint_handler,
        )

        await dependencies._agent_registry.register(awaitable_agent)

        await dependencies._tool_authorizer.allow(
            authenticated_principal,
            "search_documents",
        )

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
                        call_id=call_id,
                        name="search_documents",
                        arguments={"query": "enterprise AI"},
                    )
                ],
            },
            {
                "reply": (
                    "The MCP document search returned enterprise AI " "architecture evidence."
                ),
                "provider": "integration-test",
                "model": "integration-test-model",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 12,
                    "total_tokens": 32,
                },
                "tool_calls": [],
            },
        ]

        async def _deterministic_route_chat(request: dict) -> dict:
            assert responses, "Unexpected additional LLM request."
            return responses.pop(0)

        # The first real MCP invocation succeeds. The durable idempotency
        # completion then fails, while release is suppressed so the claim
        # remains CLAIMED. The subsequent AFTER_TOOL checkpoint also fails,
        # causing the control-plane run to become FAILED with a durable
        # BEFORE_TOOL checkpoint.
        with (
            patch(
                "app.control_plane.dependencies._llm_router.route_chat",
                new=AsyncMock(side_effect=_deterministic_route_chat),
            ) as mock_route_chat,
            patch.object(
                dependencies._agent_checkpoint_handler,
                "save",
                new=AsyncMock(side_effect=_failing_checkpoint_save),
            ),
            patch.object(
                dependencies._tool_idempotency_store,
                "complete",
                new=AsyncMock(side_effect=_failing_idempotency_complete),
            ),
            patch.object(
                dependencies._tool_idempotency_store,
                "release",
                new=AsyncMock(side_effect=_preserve_claim_on_release),
            ),
            patch.object(
                MCPPythonSDKClient,
                "call_tool",
                new=_counting_mcp_call_tool,
            ),
        ):
            response = await client.post(
                f"/api/v1/agents/{agent_name}/run",
                headers={
                    "x-api-key": API_KEY,
                },
                json={
                    "input": "Search enterprise AI architecture documents.",
                    "session_id": session_id,
                    "user_id": "integration-mcp-recovery-user",
                    "metadata": {
                        "test": "production-mcp-agent-run-recovery",
                    },
                },
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "deterministic MCP checkpoint persistence failure"

        assert mock_route_chat.await_count == 1
        assert len(responses) == 1

        assert len(mcp_invocations) == 1
        assert mcp_invocations[0]["name"] == "search_documents"
        assert mcp_invocations[0]["arguments"] == {
            "query": "enterprise AI",
        }

        external_meta = mcp_invocations[0]["meta"]

        assert external_meta is not None
        assert external_meta["deldai"]["idempotency_key"].startswith("deldai:")

        with SessionLocal() as session:
            run_record = session.scalar(
                select(AgentRunRecord)
                .where(
                    AgentRunRecord.agent_name == agent_name,
                    AgentRunRecord.session_id == session_id,
                )
                .order_by(AgentRunRecord.started_at.desc())
            )

            assert run_record is not None
            run_id = run_record.run_id

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

            idempotency_record = session.scalar(
                select(ToolExecutionIdempotencyRecord).where(
                    ToolExecutionIdempotencyRecord.run_id == run_id,
                    ToolExecutionIdempotencyRecord.call_id == call_id,
                    ToolExecutionIdempotencyRecord.tool_name == "search_documents",
                )
            )

        assert run_record.status == "failed"

        assert len(checkpoints) == 1
        assert checkpoints[0].position == AgentCheckpointPosition.BEFORE_TOOL_EXECUTION.value

        assert idempotency_record is not None
        assert idempotency_record.status == "claimed"

        # Restore the real durability behavior before invoking recovery.
        dependencies._tool_idempotency_store.complete = original_idempotency_complete
        dependencies._tool_idempotency_store.release = original_idempotency_release

        # Recovery is deliberately built from the production dependency
        # factory, including the real PostgreSQL idempotency store.
        with SessionLocal() as session:
            recovery_service = await dependencies.build_agent_run_recovery_service(session)

            with patch(
                "app.control_plane.dependencies._llm_router.route_chat",
                new=AsyncMock(side_effect=_deterministic_route_chat),
            ) as recovery_route_chat:
                recovered = await recovery_service.recover(run_id)

            assert recovery_route_chat.await_count == 1

        assert recovered is not None
        assert recovered.response.output == (
            "The MCP document search returned enterprise AI " "architecture evidence."
        )

        # Recovery must not cross the external MCP boundary again.
        assert len(mcp_invocations) == 1

        # The second LLM response was consumed only after recovery resumed
        # from BEFORE_TOOL_EXECUTION and received the durable ambiguity result.
        assert mock_route_chat.await_count == 1
        assert recovery_route_chat.await_count == 1
        assert responses == []

        with SessionLocal() as session:
            final_run = session.scalar(
                select(AgentRunRecord).where(
                    AgentRunRecord.run_id == run_id,
                )
            )

            final_idempotency_record = session.scalar(
                select(ToolExecutionIdempotencyRecord).where(
                    ToolExecutionIdempotencyRecord.run_id == run_id,
                    ToolExecutionIdempotencyRecord.call_id == call_id,
                    ToolExecutionIdempotencyRecord.tool_name == "search_documents",
                )
            )

        assert final_run is not None
        assert final_run.status == "completed"

        assert final_idempotency_record is not None
        assert final_idempotency_record.status == "ambiguous"

    finally:
        # Shut down the real MCP stdio server before restoring configuration.
        await dependencies.close_mcp_servers()
        await client.aclose()

        if original_mcp_servers is None:
            os.environ.pop("MCP_SERVERS", None)
        else:
            os.environ["MCP_SERVERS"] = original_mcp_servers

        # Restore any patched production dependency methods if the test
        # fails before the explicit restoration above.
        dependencies._agent_checkpoint_handler.save = original_checkpoint_save
        dependencies._tool_idempotency_store.complete = original_idempotency_complete
        dependencies._tool_idempotency_store.release = original_idempotency_release

        await dependencies._agent_registry.remove(agent_name)

        if run_id is not None:
            with SessionLocal() as session:
                session.query(ToolExecutionIdempotencyRecord).filter(
                    ToolExecutionIdempotencyRecord.run_id == run_id
                ).delete(synchronize_session=False)

                session.query(AgentRunCheckpointRecord).filter(
                    AgentRunCheckpointRecord.run_id == run_id
                ).delete(synchronize_session=False)

                session.query(AgentRunStepRecord).filter(
                    AgentRunStepRecord.run_id == run_id
                ).delete(synchronize_session=False)

                session.query(AgentRunEventRecord).filter(
                    AgentRunEventRecord.run_id == run_id
                ).delete(synchronize_session=False)

                session.query(AgentRunRecord).filter(AgentRunRecord.run_id == run_id).delete(
                    synchronize_session=False
                )

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
