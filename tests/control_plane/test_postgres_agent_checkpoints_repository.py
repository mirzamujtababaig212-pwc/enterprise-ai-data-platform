import pytest
from sqlalchemy import create_engine, delete, inspect
from sqlalchemy.orm import sessionmaker

from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.persistence.models import (
    AgentRunCheckpointRecord,
    AgentRunRecord,
)

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_messages import AgentMessage, AgentMessageRole
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)
from app.control_plane.agent_checkpoints.repository import (
    AgentCheckpointsRepository,
)
from app.control_plane.persistence.models import Base


def make_checkpoint(
    *,
    run_id: str = "run-1",
    tool_round: int = 1,
    metadata: dict | None = None,
) -> AgentExecutionCheckpoint:
    return AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id=run_id,
        agent_name="vehicle-agent",
        session_id="session-1",
        user_id="user-1",
        messages=(
            AgentMessage(
                role=AgentMessageRole.USER,
                content="Find the vehicle risk.",
            ),
            AgentMessage(
                role=AgentMessageRole.ASSISTANT,
                content="I need to retrieve vehicle information.",
            ),
        ),
        tool_round=tool_round,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={} if metadata is None else metadata,
    )


def make_repository():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session = session_factory()

    return engine, session, PostgreSQLAgentCheckpointsRepository(session)


def test_schema_contains_agent_run_checkpoints_table() -> None:
    engine, session, repository = make_repository()

    try:
        assert inspect(repository._session.bind).has_table("agent_run_checkpoints")
    finally:
        session.close()
        engine.dispose()


def test_save_and_get_latest_round_trip() -> None:
    engine, session, repository = make_repository()

    try:
        checkpoint = make_checkpoint(
            metadata={
                "source": "agent_execution",
                "tool": "vehicle_lookup",
            }
        )

        result = repository.save(checkpoint)
        restored = repository.get_latest(checkpoint.run_id)

        assert result == checkpoint
        assert restored is not None
        assert restored.schema_version == checkpoint.schema_version
        assert restored.run_id == checkpoint.run_id
        assert restored.agent_name == checkpoint.agent_name
        assert restored.session_id == checkpoint.session_id
        assert restored.user_id == checkpoint.user_id
        assert restored.messages == checkpoint.messages
        assert restored.tool_round == checkpoint.tool_round
        assert restored.position == checkpoint.position
        assert restored.metadata == checkpoint.metadata
        assert (
            restored.execution_budget_state.llm_calls == checkpoint.execution_budget_state.llm_calls
        )
        assert (
            restored.execution_budget_state.tool_calls
            == checkpoint.execution_budget_state.tool_calls
        )
        assert (
            restored.execution_budget_state.tool_rounds
            == checkpoint.execution_budget_state.tool_rounds
        )
    finally:
        session.close()
        engine.dispose()


def test_get_latest_missing_run_returns_none() -> None:
    engine, session, repository = make_repository()

    try:
        assert repository.get_latest("does-not-exist") is None
    finally:
        session.close()
        engine.dispose()


def test_save_can_leave_transaction_uncommitted() -> None:
    engine, session, repository = make_repository()

    try:
        checkpoint = make_checkpoint()

        repository.save(checkpoint, commit=False)

        restored = repository.get_latest(checkpoint.run_id)
        assert restored is not None
        assert restored.run_id == checkpoint.run_id
        assert restored.messages == checkpoint.messages
        assert (
            restored.execution_budget_state.llm_calls == checkpoint.execution_budget_state.llm_calls
        )
        assert (
            restored.execution_budget_state.tool_calls
            == checkpoint.execution_budget_state.tool_calls
        )
        assert (
            restored.execution_budget_state.tool_rounds
            == checkpoint.execution_budget_state.tool_rounds
        )

        session.rollback()

        assert repository.get_latest(checkpoint.run_id) is None
    finally:
        session.close()
        engine.dispose()


def test_get_latest_returns_newest_checkpoint() -> None:
    engine, session, repository = make_repository()

    try:
        first = make_checkpoint(
            tool_round=1,
            metadata={"sequence": 1},
        )
        second = make_checkpoint(
            tool_round=2,
            metadata={"sequence": 2},
        )

        repository.save(first)
        repository.save(second)

        restored = repository.get_latest("run-1")

        assert restored is not None
        assert restored.run_id == second.run_id
        assert restored.messages == second.messages
        assert restored.tool_round == second.tool_round
        assert restored.metadata == second.metadata
        assert restored.execution_budget_state.llm_calls == second.execution_budget_state.llm_calls
        assert (
            restored.execution_budget_state.tool_calls == second.execution_budget_state.tool_calls
        )
        assert (
            restored.execution_budget_state.tool_rounds == second.execution_budget_state.tool_rounds
        )
    finally:
        session.close()
        engine.dispose()


def test_get_latest_filters_by_run_id() -> None:
    engine, session, repository = make_repository()

    try:
        repository.save(make_checkpoint(run_id="run-1"))
        repository.save(make_checkpoint(run_id="run-2"))

        restored = repository.get_latest("run-1")

        assert restored is not None
        assert restored.run_id == "run-1"
    finally:
        session.close()
        engine.dispose()


def test_repository_implements_contract() -> None:
    engine, session, repository = make_repository()

    try:
        contract: AgentCheckpointsRepository = repository

        assert contract is not None
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def postgres_repository():
    import os

    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}")

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )
    session = session_factory()

    try:
        yield engine, session, PostgreSQLAgentCheckpointsRepository(session)
    finally:
        session.close()
        engine.dispose()


def test_postgres_save_and_get_latest_round_trip(postgres_repository) -> None:
    _, session, checkpoint_repository = postgres_repository
    run_id = "checkpoint-postgres-round-trip"
    run_repository = PostgreSQLAgentRunRepository(session)

    try:
        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="vehicle-agent",
                session_id="session-1",
                user_id="user-1",
                status=AgentRunStatus.RUNNING,
            )
        )

        checkpoint = make_checkpoint(
            run_id=run_id,
            metadata={
                "source": "postgres_integration",
                "tool": "vehicle_lookup",
            },
        )

        result = checkpoint_repository.save(checkpoint)
        restored = checkpoint_repository.get_latest(run_id)

        assert result == checkpoint
        assert restored is not None
        assert restored.schema_version == checkpoint.schema_version
        assert restored.run_id == checkpoint.run_id
        assert restored.agent_name == checkpoint.agent_name
        assert restored.session_id == checkpoint.session_id
        assert restored.user_id == checkpoint.user_id
        assert restored.messages == checkpoint.messages
        assert restored.tool_round == checkpoint.tool_round
        assert restored.position == checkpoint.position
        assert restored.metadata == checkpoint.metadata
        assert (
            restored.execution_budget_state.llm_calls == checkpoint.execution_budget_state.llm_calls
        )
        assert (
            restored.execution_budget_state.tool_calls
            == checkpoint.execution_budget_state.tool_calls
        )
        assert (
            restored.execution_budget_state.tool_rounds
            == checkpoint.execution_budget_state.tool_rounds
        )
    finally:
        session.execute(
            delete(AgentRunCheckpointRecord).where(AgentRunCheckpointRecord.run_id == run_id)
        )
        session.execute(delete(AgentRunRecord).where(AgentRunRecord.run_id == run_id))
        session.commit()


def test_postgres_get_latest_returns_newest_checkpoint(
    postgres_repository,
) -> None:
    _, session, checkpoint_repository = postgres_repository
    run_id = "checkpoint-postgres-newest"
    run_repository = PostgreSQLAgentRunRepository(session)

    try:
        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="vehicle-agent",
                session_id="session-1",
                user_id="user-1",
                status=AgentRunStatus.RUNNING,
            )
        )

        first = make_checkpoint(
            run_id=run_id,
            tool_round=1,
            metadata={"sequence": 1},
        )
        second = make_checkpoint(
            run_id=run_id,
            tool_round=2,
            metadata={"sequence": 2},
        )

        checkpoint_repository.save(first)
        checkpoint_repository.save(second)

        restored = checkpoint_repository.get_latest(run_id)

        assert restored is not None
        assert restored.run_id == second.run_id
        assert restored.messages == second.messages
        assert restored.tool_round == second.tool_round
        assert restored.metadata == second.metadata
        assert restored.execution_budget_state.llm_calls == second.execution_budget_state.llm_calls
        assert (
            restored.execution_budget_state.tool_calls == second.execution_budget_state.tool_calls
        )
        assert (
            restored.execution_budget_state.tool_rounds == second.execution_budget_state.tool_rounds
        )
    finally:
        session.execute(
            delete(AgentRunCheckpointRecord).where(AgentRunCheckpointRecord.run_id == run_id)
        )
        session.execute(delete(AgentRunRecord).where(AgentRunRecord.run_id == run_id))
        session.commit()
