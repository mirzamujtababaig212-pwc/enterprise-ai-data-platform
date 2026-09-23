import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, delete, inspect
from sqlalchemy.orm import sessionmaker

from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_steps.postgres_repository import (
    PostgreSQLAgentRunStepsRepository,
)
from app.control_plane.agent_runs.models import AgentRunStatus
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.persistence.models import (
    AgentRunRecord,
    AgentRunStepRecord,
)


def make_run(
    *,
    run_id: str,
    agent_name: str = "postgres-step-integration-agent",
):
    from app.control_plane.agent_runs.models import AgentRun

    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        session_id=f"session-{run_id}",
        user_id="integration-user",
        principal="integration-principal",
        status=AgentRunStatus.PENDING,
        recovery_attempts=0,
        metadata={},
    )


def make_step(
    *,
    run_id: str,
    step_id: str,
    step_index: int = 0,
    step_type: str = "tool",
    status: AgentRunStepStatus = AgentRunStepStatus.PLANNED,
    attempt: int = 1,
    tool_name: str | None = "vehicle_query",
    call_id: str | None = "call-1",
    input=None,
    metadata: dict | None = None,
) -> AgentRunStep:
    return AgentRunStep(
        run_id=run_id,
        step_id=step_id,
        step_index=step_index,
        step_type=step_type,
        status=status,
        attempt=attempt,
        tool_name=tool_name,
        call_id=call_id,
        input=({"vehicle_id": "V-100"} if input is None else input),
        metadata={} if metadata is None else metadata,
    )


@pytest.fixture()
def postgres_sessions():
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

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    try:
        yield engine, session_factory
    finally:
        engine.dispose()


def create_parent_run(
    session_factory,
    *,
    run_id: str,
) -> None:
    session = session_factory()
    try:
        repository = PostgreSQLAgentRunRepository(session)

        existing = repository.get(run_id)

        if existing is None:
            repository.create(make_run(run_id=run_id))
    finally:
        session.close()


def clear_integration_run(
    session_factory,
    *,
    run_id: str,
) -> None:
    session = session_factory()
    try:
        session.execute(
            delete(AgentRunStepRecord).where(
                AgentRunStepRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
    finally:
        session.close()


@pytest.mark.asyncio
async def test_postgres_agent_run_steps_table_exists(
    postgres_sessions,
) -> None:
    engine, _ = postgres_sessions

    assert inspect(engine).has_table("agent_run_steps")


@pytest.mark.asyncio
async def test_postgres_step_round_trip_persists_jsonb_and_metadata(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "postgres-step-round-trip"
    clear_integration_run(session_factory, run_id=run_id)
    create_parent_run(session_factory, run_id=run_id)

    session = session_factory()
    repository = PostgreSQLAgentRunStepsRepository(session)

    try:
        step = make_step(
            run_id=run_id,
            step_id="step-round-trip",
            input={
                "vehicle_id": "V-100",
                "filters": {"status": "active"},
            },
            metadata={
                "trace_id": "trace-123",
                "source": "integration-test",
            },
        )

        created = repository.create(step)
        restored = repository.get(run_id, "step-round-trip")

        assert created.run_id == run_id
        assert restored is not None
        assert restored.step_id == "step-round-trip"
        assert restored.input == {
            "vehicle_id": "V-100",
            "filters": {"status": "active"},
        }
        assert restored.metadata == {
            "trace_id": "trace-123",
            "source": "integration-test",
        }
        assert restored.status is AgentRunStepStatus.PLANNED
        assert restored.attempt == 1
    finally:
        session.close()
        clear_integration_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_postgres_step_lifecycle_and_retry_are_durable(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "postgres-step-lifecycle"
    clear_integration_run(session_factory, run_id=run_id)
    create_parent_run(session_factory, run_id=run_id)

    session = session_factory()
    repository = PostgreSQLAgentRunStepsRepository(session)

    try:
        repository.create(
            make_step(
                run_id=run_id,
                step_id="step-lifecycle",
                step_index=1,
            )
        )

        started_at = datetime.now(UTC)

        running = repository.transition(
            run_id,
            "step-lifecycle",
            status=AgentRunStepStatus.RUNNING,
            updated_at=started_at,
            started_at=started_at,
        )

        assert running is not None
        assert running.status is AgentRunStepStatus.RUNNING
        assert running.started_at is not None
        assert running.completed_at is None

        failed_at = datetime.now(UTC)

        failed = repository.transition(
            run_id,
            "step-lifecycle",
            status=AgentRunStepStatus.FAILED,
            updated_at=failed_at,
            completed_at=failed_at,
            error="vehicle query failed",
            failure_category="execution_error",
        )

        assert failed is not None
        assert failed.status is AgentRunStepStatus.FAILED
        assert failed.error == "vehicle query failed"
        assert failed.failure_category == "execution_error"

        retry_started_at = datetime.now(UTC)

        retried = repository.retry(
            run_id,
            "step-lifecycle",
            updated_at=retry_started_at,
            started_at=retry_started_at,
        )

        assert retried is not None
        assert retried.status is AgentRunStepStatus.RUNNING
        assert retried.attempt == 2
        assert retried.error is None
        assert retried.failure_category is None
        assert retried.completed_at is None

        completed_at = datetime.now(UTC)

        completed = repository.transition(
            run_id,
            "step-lifecycle",
            status=AgentRunStepStatus.COMPLETED,
            updated_at=completed_at,
            completed_at=completed_at,
            output={
                "rows": 4,
                "source": "delta",
            },
        )

        assert completed is not None
        assert completed.status is AgentRunStepStatus.COMPLETED
        assert completed.output == {
            "rows": 4,
            "source": "delta",
        }

        session.close()

        restarted_session = session_factory()
        restarted_repository = PostgreSQLAgentRunStepsRepository(restarted_session)

        try:
            restored = restarted_repository.get(
                run_id,
                "step-lifecycle",
            )

            assert restored is not None
            assert restored.status is AgentRunStepStatus.COMPLETED
            assert restored.attempt == 2
            assert restored.output == {
                "rows": 4,
                "source": "delta",
            }
            assert restored.error is None
            assert restored.failure_category is None
        finally:
            restarted_session.close()
    finally:
        if session.is_active:
            session.close()
        clear_integration_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_postgres_steps_are_ordered_and_filterable_by_status(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "postgres-step-ordering"
    clear_integration_run(session_factory, run_id=run_id)
    create_parent_run(session_factory, run_id=run_id)

    session = session_factory()
    repository = PostgreSQLAgentRunStepsRepository(session)

    try:
        repository.create(
            make_step(
                run_id=run_id,
                step_id="step-3",
                step_index=3,
            )
        )
        repository.create(
            make_step(
                run_id=run_id,
                step_id="step-1",
                step_index=1,
                status=AgentRunStepStatus.COMPLETED,
            )
        )
        repository.create(
            make_step(
                run_id=run_id,
                step_id="step-2",
                step_index=2,
                status=AgentRunStepStatus.FAILED,
            )
        )

        steps = repository.list(run_id)

        assert [step.step_id for step in steps] == [
            "step-1",
            "step-2",
            "step-3",
        ]

        failed = repository.list(
            run_id,
            status=AgentRunStepStatus.FAILED,
        )

        assert [step.step_id for step in failed] == ["step-2"]
    finally:
        session.close()
        clear_integration_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_postgres_duplicate_step_is_rejected(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "postgres-step-duplicate"
    clear_integration_run(session_factory, run_id=run_id)
    create_parent_run(session_factory, run_id=run_id)

    session = session_factory()
    repository = PostgreSQLAgentRunStepsRepository(session)

    try:
        step = make_step(
            run_id=run_id,
            step_id="duplicate-step",
        )

        repository.create(step)

        with pytest.raises(Exception):
            repository.create(step)
    finally:
        session.close()
        clear_integration_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_postgres_commit_false_rolls_back_step_creation(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "postgres-step-rollback"
    clear_integration_run(session_factory, run_id=run_id)
    create_parent_run(session_factory, run_id=run_id)

    session = session_factory()
    repository = PostgreSQLAgentRunStepsRepository(session)

    try:
        repository.create(
            make_step(
                run_id=run_id,
                step_id="rollback-step",
            ),
            commit=False,
        )

        session.rollback()

        restored = repository.get(
            run_id,
            "rollback-step",
        )

        assert restored is None
    finally:
        session.close()
        clear_integration_run(session_factory, run_id=run_id)


@pytest.mark.asyncio
async def test_postgres_ambiguous_step_is_terminal_and_durable(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    run_id = "postgres-step-ambiguous"
    clear_integration_run(session_factory, run_id=run_id)
    create_parent_run(session_factory, run_id=run_id)

    session = session_factory()
    repository = PostgreSQLAgentRunStepsRepository(session)

    try:
        repository.create(
            make_step(
                run_id=run_id,
                step_id="ambiguous-step",
            )
        )

        started_at = datetime.now(UTC)

        repository.transition(
            run_id,
            "ambiguous-step",
            status=AgentRunStepStatus.RUNNING,
            updated_at=started_at,
            started_at=started_at,
        )

        ambiguous_at = datetime.now(UTC)

        ambiguous = repository.transition(
            run_id,
            "ambiguous-step",
            status=AgentRunStepStatus.AMBIGUOUS,
            updated_at=ambiguous_at,
            completed_at=ambiguous_at,
            error="external execution outcome cannot be proven",
            failure_category="ambiguous_execution",
        )

        assert ambiguous is not None
        assert ambiguous.status is AgentRunStepStatus.AMBIGUOUS

        session.close()

        restarted_session = session_factory()
        restarted_repository = PostgreSQLAgentRunStepsRepository(restarted_session)

        try:
            restored = restarted_repository.get(
                run_id,
                "ambiguous-step",
            )

            assert restored is not None
            assert restored.status is AgentRunStepStatus.AMBIGUOUS
            assert restored.error == ("external execution outcome cannot be proven")

            with pytest.raises(Exception):
                restarted_repository.transition(
                    run_id,
                    "ambiguous-step",
                    status=AgentRunStepStatus.RUNNING,
                    updated_at=datetime.now(UTC),
                    started_at=datetime.now(UTC),
                )
        finally:
            restarted_session.close()
    finally:
        if session.is_active:
            session.close()
        clear_integration_run(session_factory, run_id=run_id)
