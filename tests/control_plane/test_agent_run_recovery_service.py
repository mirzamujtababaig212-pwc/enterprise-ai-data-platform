from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_messages import user_message
from ai_platform.agents.models import (
    AgentRequest,
    AgentResponse,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.recovery_service import (
    AgentRunRecoveryService,
)
from app.control_plane.agent_runs.request_snapshot import (
    AgentRunRequestSnapshot,
)
from app.control_plane.agent_runs.in_memory import (
    InMemoryAgentRunRepository,
)


def failed_run(
    *,
    run_id: str = "run-123",
    agent_name: str = "recoverable-agent",
    request_snapshot: AgentRunRequestSnapshot | None = None,
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        session_id="session-123",
        user_id="user-123",
        status=AgentRunStatus.FAILED,
        completed_at=datetime.now(UTC),
        error_type="RuntimeError",
        error_message="original failure",
        metadata={"request_id": "req-123"},
        request_snapshot=request_snapshot,
    )


def request_snapshot() -> AgentRunRequestSnapshot:
    return AgentRunRequestSnapshot.from_request(
        AgentRequest(
            input="Find vehicle incidents for fleet-42",
            session_id="session-123",
            user_id="user-123",
            memory_namespace="fleet-memory",
            metadata={
                "request_id": "req-123",
                "source": "api",
            },
        )
    )


def checkpoint() -> AgentExecutionCheckpoint:
    return AgentExecutionCheckpoint(
        schema_version=1,
        run_id="run-123",
        agent_name="recoverable-agent",
        session_id="session-123",
        user_id="user-123",
        messages=(user_message("Find vehicle incidents for fleet-42"),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={},
    )


class FakeCheckpointRepository:
    def __init__(
        self,
        value: AgentExecutionCheckpoint | None,
    ) -> None:
        self.value = value
        self.requested_run_id = None

    def save(self, checkpoint, *, commit=True):
        return checkpoint

    def get_latest(self, run_id):
        self.requested_run_id = run_id
        return self.value


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
            output="Recovered successfully.",
            session_id=request.session_id,
        )


@pytest.mark.asyncio
async def test_recovery_claims_failed_run_and_resumes_from_checkpoint():
    repository = InMemoryAgentRunRepository()

    run = failed_run(
        request_snapshot=request_snapshot(),
    )
    repository.create(run)

    checkpoint_repository = FakeCheckpointRepository(checkpoint())
    runtime = FakeRuntime()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=checkpoint_repository,
    )

    result = await service.recover("run-123")

    assert isinstance(result, AgentRunExecutionResult)
    assert result.run_id == "run-123"
    assert result.response.output == "Recovered successfully."

    recovered = repository.get("run-123")

    assert recovered is not None
    assert recovered.status is AgentRunStatus.COMPLETED
    assert recovered.output == "Recovered successfully."
    assert recovered.error_type is None
    assert recovered.error_message is None

    assert len(runtime.calls) == 1

    call = runtime.calls[0]

    assert call["agent_name"] == "recoverable-agent"
    assert call["run_id"] == "run-123"
    assert call["checkpoint"] is checkpoint_repository.value

    restored_request = call["request"]

    assert restored_request.input == "Find vehicle incidents for fleet-42"
    assert restored_request.session_id == "session-123"
    assert restored_request.user_id == "user-123"
    assert restored_request.memory_namespace == "fleet-memory"
    assert restored_request.metadata == {
        "request_id": "req-123",
        "source": "api",
    }


@pytest.mark.asyncio
async def test_recovery_does_not_recover_run_without_request_snapshot():
    repository = InMemoryAgentRunRepository()

    repository.create(
        failed_run(
            request_snapshot=None,
        )
    )

    runtime = FakeRuntime()
    checkpoint_repository = FakeCheckpointRepository(checkpoint())

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=checkpoint_repository,
    )

    with pytest.raises(
        RuntimeError,
        match="has no request snapshot",
    ):
        await service.recover("run-123")

    run = repository.get("run-123")

    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_recovery_marks_run_failed_when_checkpoint_is_missing():
    repository = InMemoryAgentRunRepository()

    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        )
    )

    runtime = FakeRuntime()
    checkpoint_repository = FakeCheckpointRepository(None)

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=checkpoint_repository,
    )

    with pytest.raises(
        RuntimeError,
        match="has no execution checkpoint",
    ):
        await service.recover("run-123")

    run = repository.get("run-123")

    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.error_type == "RuntimeError"
    assert "has no execution checkpoint" in run.error_message
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_recovery_marks_run_failed_when_runtime_resume_fails():
    repository = InMemoryAgentRunRepository()

    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        )
    )

    checkpoint_repository = FakeCheckpointRepository(checkpoint())

    class FailingRuntime:
        async def resume(
            self,
            agent_name,
            request,
            checkpoint,
            *,
            run_id=None,
        ):
            raise ValueError("LLM provider unavailable")

    service = AgentRunRecoveryService(
        runtime=FailingRuntime(),
        repository=repository,
        checkpoints_repository=checkpoint_repository,
    )

    with pytest.raises(
        ValueError,
        match="LLM provider unavailable",
    ):
        await service.recover("run-123")

    run = repository.get("run-123")

    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.error_type == "ValueError"
    assert run.error_message == "LLM provider unavailable"


@pytest.mark.asyncio
async def test_recovery_rejects_non_failed_run():
    repository = InMemoryAgentRunRepository()

    run = failed_run(
        request_snapshot=request_snapshot(),
    ).model_copy(
        update={
            "status": AgentRunStatus.COMPLETED,
        }
    )

    repository.create(run)

    service = AgentRunRecoveryService(
        runtime=FakeRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
    )

    with pytest.raises(
        ValueError,
        match="not eligible for recovery",
    ):
        await service.recover("run-123")


@pytest.mark.asyncio
async def test_recovery_rejects_missing_run():
    repository = InMemoryAgentRunRepository()

    service = AgentRunRecoveryService(
        runtime=FakeRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
    )

    with pytest.raises(
        LookupError,
        match="was not found",
    ):
        await service.recover("missing-run")


@pytest.mark.asyncio
async def test_recovery_claim_prevents_second_recovery():
    repository = InMemoryAgentRunRepository()

    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        )
    )

    runtime = FakeRuntime()
    checkpoint_repository = FakeCheckpointRepository(checkpoint())

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=checkpoint_repository,
    )

    first = await service.recover("run-123")

    assert first.response.output == "Recovered successfully."

    with pytest.raises(
        ValueError,
        match="not eligible for recovery",
    ):
        await service.recover("run-123")

    assert len(runtime.calls) == 1
