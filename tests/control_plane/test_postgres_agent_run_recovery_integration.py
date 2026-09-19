from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_messages import user_message
from ai_platform.agents.models import AgentRequest, AgentResponse
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunStatus,
)
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.agent_runs.recovery_service import (
    AgentRunRecoveryService,
)
from app.control_plane.agent_runs.request_snapshot import (
    AgentRunRequestSnapshot,
)
from app.control_plane.persistence.models import (
    AgentRunCheckpointRecord,
    AgentRunRecord,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = []

    async def resume(
        self,
        agent_name,
        request,
        checkpoint,
        *,
        run_id=None,
    ):
        self.calls.append(
            {
                "agent_name": agent_name,
                "request": request,
                "checkpoint": checkpoint,
                "run_id": run_id,
            }
        )

        return AgentResponse(
            agent_name=agent_name,
            output="Recovered from PostgreSQL checkpoint.",
            session_id=request.session_id,
        )


def make_engine():
    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )


def make_checkpoint(run_id: str) -> AgentExecutionCheckpoint:
    return AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id=run_id,
        agent_name="recoverable-agent",
        session_id="session-postgres",
        user_id="user-postgres",
        messages=(user_message("Find vehicle incidents for fleet-42"),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={
            "source": "postgres_recovery_integration",
        },
    )


def make_request_snapshot() -> AgentRunRequestSnapshot:
    return AgentRunRequestSnapshot.from_request(
        AgentRequest(
            input="Find vehicle incidents for fleet-42",
            session_id="session-postgres",
            user_id="user-postgres",
            memory_namespace="fleet-memory",
            metadata={
                "request_id": "postgres-recovery-request",
                "source": "integration-test",
            },
        )
    )


def test_postgres_stale_run_recovers_from_persisted_checkpoint() -> None:
    engine = make_engine()
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-stale-recovery-e2e"
    stale_before = datetime.now(UTC)
    expired_at = stale_before - timedelta(minutes=5)

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        checkpoint_repository = PostgreSQLAgentCheckpointsRepository(session)

        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="recoverable-agent",
                session_id="session-postgres",
                user_id="user-postgres",
                status=AgentRunStatus.RUNNING,
                started_at=expired_at - timedelta(minutes=1),
                lease_id="expired-lease",
                lease_expires_at=expired_at,
                request_snapshot=make_request_snapshot(),
            )
        )

        checkpoint = make_checkpoint(run_id)
        checkpoint_repository.save(checkpoint)

        runtime = FakeRuntime()

        service = AgentRunRecoveryService(
            runtime=runtime,
            repository=run_repository,
            checkpoints_repository=checkpoint_repository,
            lease_seconds=60,
        )

        results = __import__("asyncio").run(
            service.recover_stale_runs(
                stale_before=stale_before,
                limit=10,
            )
        )

        assert len(results) == 1
        assert results[0].run_id == run_id
        assert results[0].response.output == ("Recovered from PostgreSQL checkpoint.")

        assert len(runtime.calls) == 1

        call = runtime.calls[0]

        assert call["run_id"] == run_id
        assert call["agent_name"] == "recoverable-agent"
        assert call["checkpoint"] == checkpoint

        request = call["request"]

        assert request.input == "Find vehicle incidents for fleet-42"
        assert request.session_id == "session-postgres"
        assert request.user_id == "user-postgres"
        assert request.memory_namespace == "fleet-memory"
        assert request.metadata == {
            "request_id": "postgres-recovery-request",
            "source": "integration-test",
        }

        session.expire_all()

        restored = run_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.COMPLETED
        assert restored.output == "Recovered from PostgreSQL checkpoint."
        assert restored.lease_id is None
        assert restored.lease_expires_at is None
        assert restored.error_type is None
        assert restored.error_message is None

        restored_checkpoint = checkpoint_repository.get_latest(run_id)

        assert restored_checkpoint == checkpoint
    finally:
        session.rollback()
        session.execute(
            delete(AgentRunCheckpointRecord).where(
                AgentRunCheckpointRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
        session.close()
        engine.dispose()
