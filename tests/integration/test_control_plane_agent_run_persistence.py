"""End-to-end control-plane agent run persistence integration test."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.control_plane.agent_runs.models import AgentRunStatus
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
