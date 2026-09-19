import pytest

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_messages import AgentMessage, AgentMessageRole
from app.control_plane.agent_checkpoints.postgres_handler import (
    PostgreSQLAgentCheckpointHandler,
)


def make_checkpoint() -> AgentExecutionCheckpoint:
    return AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id="run-1",
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
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={"source": "agent_execution"},
    )


class FakeSession:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeRepository:
    def __init__(self, session) -> None:
        self.session = session
        self.saved = []

    def save(self, checkpoint):
        self.saved.append(checkpoint)
        return checkpoint


@pytest.mark.asyncio
async def test_handler_persists_checkpoint_and_closes_session(monkeypatch):
    session = FakeSession()
    repository = FakeRepository(session)

    monkeypatch.setattr(
        "app.control_plane.agent_checkpoints.postgres_handler."
        "PostgreSQLAgentCheckpointsRepository",
        lambda supplied_session: repository,
    )

    handler = PostgreSQLAgentCheckpointHandler(lambda: session)
    checkpoint = make_checkpoint()

    await handler.save(checkpoint)

    assert repository.saved == [checkpoint]
    assert session.closed is True


@pytest.mark.asyncio
async def test_handler_does_not_raise_when_persistence_fails(monkeypatch):
    session = FakeSession()

    class FailingRepository:
        def __init__(self, supplied_session) -> None:
            pass

        def save(self, checkpoint):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "app.control_plane.agent_checkpoints.postgres_handler."
        "PostgreSQLAgentCheckpointsRepository",
        FailingRepository,
    )

    handler = PostgreSQLAgentCheckpointHandler(lambda: session)

    await handler.save(make_checkpoint())

    assert session.closed is True
