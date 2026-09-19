from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

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
        assert restored == checkpoint
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

        assert repository.get_latest(checkpoint.run_id) == checkpoint

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

        assert restored == second
        assert restored is not None
        assert restored.tool_round == 2
        assert restored.metadata == {"sequence": 2}
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
