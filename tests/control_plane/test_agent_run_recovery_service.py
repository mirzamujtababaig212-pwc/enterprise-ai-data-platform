from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from ai_platform.agents.exceptions import AgentExecutionOwnershipLostError
from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_messages import user_message
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.models import (
    AgentRequest,
    AgentResponse,
)
from ai_platform.agents.policy import (
    PolicyViolationError,
    TenantPolicy,
    TenantPolicyEngine,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.cancellation import (
    AgentRunCancellationRegistry,
)
from app.control_plane.agent_runs.in_memory import (
    InMemoryAgentRunRepository,
)
from app.control_plane.agent_runs.exceptions import RecoveryExhaustedError
from app.control_plane.agent_runs.recovery_service import (
    AgentRunRecoveryService,
)
from app.control_plane.agent_runs.request_snapshot import (
    AgentRunRequestSnapshot,
)


def failed_run(
    *,
    run_id: str = "run-123",
    agent_name: str = "recoverable-agent",
    request_snapshot: AgentRunRequestSnapshot | None = None,
    recovery_attempts: int = 0,
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
        recovery_attempts=recovery_attempts,
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


class RecordingRepository(InMemoryAgentRunRepository):
    def __init__(self) -> None:
        super().__init__()
        self.recovery_claim = None
        self.stale_recovery_claim = None
        self.complete_call = None
        self.fail_call = None

    def claim_for_recovery(
        self,
        run_id,
        *,
        started_at,
        lease_id,
        lease_expires_at,
        max_recovery_attempts,
    ):
        self.recovery_claim = {
            "run_id": run_id,
            "started_at": started_at,
            "lease_id": lease_id,
            "lease_expires_at": lease_expires_at,
            "max_recovery_attempts": max_recovery_attempts,
        }
        return super().claim_for_recovery(
            run_id,
            started_at=started_at,
            lease_id=lease_id,
            lease_expires_at=lease_expires_at,
            max_recovery_attempts=max_recovery_attempts,
        )

    def claim_expired_running_run(
        self,
        run_id,
        *,
        stale_before,
        started_at,
        lease_id,
        lease_expires_at,
        max_recovery_attempts,
    ):
        self.stale_recovery_claim = {
            "run_id": run_id,
            "stale_before": stale_before,
            "started_at": started_at,
            "lease_id": lease_id,
            "lease_expires_at": lease_expires_at,
            "max_recovery_attempts": max_recovery_attempts,
        }
        return super().claim_expired_running_run(
            run_id,
            stale_before=stale_before,
            started_at=started_at,
            lease_id=lease_id,
            lease_expires_at=lease_expires_at,
            max_recovery_attempts=max_recovery_attempts,
        )

    def complete_if_owner(
        self,
        run_id,
        *,
        lease_id,
        completed_at,
        output,
    ):
        self.complete_call = {
            "run_id": run_id,
            "lease_id": lease_id,
            "completed_at": completed_at,
            "output": output,
        }
        return super().complete_if_owner(
            run_id,
            lease_id=lease_id,
            completed_at=completed_at,
            output=output,
        )

    def fail_if_owner(
        self,
        run_id,
        *,
        lease_id,
        completed_at,
        error_type,
        error_message,
    ):
        self.fail_call = {
            "run_id": run_id,
            "lease_id": lease_id,
            "completed_at": completed_at,
            "error_type": error_type,
            "error_message": error_message,
        }
        return super().fail_if_owner(
            run_id,
            lease_id=lease_id,
            completed_at=completed_at,
            error_type=error_type,
            error_message=error_message,
        )


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(self, event: AgentExecutionEvent) -> None:
        self.events.append(event)


class FailingObserver:
    async def record(self, event: AgentExecutionEvent) -> None:
        raise RuntimeError("event persistence unavailable")


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
        lease_id=None,
        execution_ownership_lost=None,
    ):
        self.calls.append(
            {
                "agent_name": agent_name,
                "request": request,
                "checkpoint": checkpoint,
                "run_id": run_id,
                "execution_ownership_lost": execution_ownership_lost,
            }
        )

        return AgentResponse(
            agent_name=agent_name,
            output="Recovered successfully.",
            session_id=request.session_id,
        )


class CancellationBlockingRuntime:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = False

    async def resume(
        self,
        agent_name,
        request,
        checkpoint,
        *,
        run_id=None,
        lease_id=None,
        execution_ownership_lost=None,
    ):
        self.started.set()

        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


@pytest.mark.asyncio
async def test_recovery_cancellation_interrupts_active_recovery_and_cancels_owned_run():
    repository = RecordingRepository()

    run = failed_run(
        request_snapshot=request_snapshot(),
    )
    repository.create(run)

    checkpoint_repository = FakeCheckpointRepository(checkpoint())
    runtime = CancellationBlockingRuntime()
    cancellation_registry = AgentRunCancellationRegistry()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=checkpoint_repository,
        cancellation_registry=cancellation_registry,
    )

    recovery_task = asyncio.create_task(
        service.recover("run-123"),
    )

    await runtime.started.wait()

    registered_task = cancellation_registry._tasks.get("run-123")
    assert registered_task is recovery_task

    cancelled = cancellation_registry.cancel("run-123")

    assert cancelled is True

    with pytest.raises(asyncio.CancelledError):
        await recovery_task

    assert runtime.cancelled is True

    recovered = repository.get("run-123")

    assert recovered is not None
    assert recovered.status is AgentRunStatus.CANCELLED
    assert recovered.cancellation_requested is False

    assert repository.complete_call is None
    assert repository.fail_call is None

    assert "run-123" not in cancellation_registry._tasks


@pytest.mark.asyncio
async def test_recovery_cancellation_does_not_mutate_run_after_lease_loss():
    class LeaseLostRepository(RecordingRepository):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_if_owner_called = False

        def cancel_if_owner(
            self,
            run_id,
            *,
            lease_id,
            completed_at,
        ):
            self.cancel_if_owner_called = True
            return None

    repository = LeaseLostRepository()

    run = failed_run(
        request_snapshot=request_snapshot(),
    )
    repository.create(run)

    checkpoint_repository = FakeCheckpointRepository(checkpoint())
    runtime = CancellationBlockingRuntime()
    cancellation_registry = AgentRunCancellationRegistry()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=checkpoint_repository,
        cancellation_registry=cancellation_registry,
    )

    recovery_task = asyncio.create_task(
        service.recover("run-123"),
    )

    await runtime.started.wait()

    registered_task = cancellation_registry._tasks.get("run-123")
    assert registered_task is recovery_task

    cancelled = cancellation_registry.cancel("run-123")

    assert cancelled is True

    with pytest.raises(asyncio.CancelledError):
        await recovery_task

    assert runtime.cancelled is True
    assert repository.cancel_if_owner_called is True
    assert repository.complete_call is None
    assert repository.fail_call is None

    recovered = repository.get("run-123")

    assert recovered is not None
    assert recovered.status is AgentRunStatus.RUNNING
    assert "run-123" not in cancellation_registry._tasks


@pytest.mark.asyncio
async def test_recovery_claims_failed_run_and_resumes_from_checkpoint():
    repository = RecordingRepository()

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

    assert repository.recovery_claim is not None
    assert repository.recovery_claim["lease_id"] is not None
    assert repository.recovery_claim["lease_expires_at"] is not None
    assert repository.recovery_claim["lease_expires_at"] > repository.recovery_claim["started_at"]

    assert repository.complete_call is not None
    assert repository.complete_call["lease_id"] == repository.recovery_claim["lease_id"]

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
async def test_recovery_rejects_memory_namespace_removed_by_current_tenant_policy():
    repository = RecordingRepository()

    request = AgentRequest(
        input="Find vehicle incidents for fleet-42",
        session_id="session-123",
        user_id="user-123",
        principal="api_key:recovery-principal",
        tenant_id="tenant-acme",
        memory_namespace="fleet-memory",
    )

    repository.create(
        failed_run(
            request_snapshot=AgentRunRequestSnapshot.from_request(request),
        )
    )

    policy_engine = TenantPolicyEngine()
    policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_memory_namespaces=frozenset({"different-memory"}),
        )
    )

    runtime = FakeRuntime()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        tenant_policy_engine=policy_engine,
    )

    with pytest.raises(
        PolicyViolationError,
        match="Memory namespace.*not allowed",
    ):
        await service.recover("run-123")

    assert runtime.calls == []

    recovered = repository.get("run-123")

    assert recovered is not None
    assert recovered.status is AgentRunStatus.FAILED
    assert recovered.error_type == "PolicyViolationError"
    assert "fleet-memory" in recovered.error_message


@pytest.mark.asyncio
async def test_recovery_emits_started_and_completed_events_for_failed_run():
    repository = InMemoryAgentRunRepository()
    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        )
    )

    observer = RecordingObserver()

    service = AgentRunRecoveryService(
        runtime=FakeRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
    )

    await service.recover("run-123")

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_RECOVERY_STARTED,
        AgentExecutionEventType.AGENT_RECOVERY_COMPLETED,
    ]

    assert observer.events[0].metadata == {
        "recovery_type": "failed_run",
        "recovery_attempt": 1,
    }
    assert observer.events[1].metadata["recovery_type"] == "failed_run"
    assert observer.events[1].metadata["recovery_attempt"] == 1
    assert isinstance(observer.events[1].metadata["duration_ms"], (int, float))
    assert observer.events[1].metadata["duration_ms"] >= 0

    for event in observer.events:
        assert event.run_id == "run-123"
        assert event.agent_name == "recoverable-agent"
        assert event.session_id == "session-123"
        assert event.user_id == "user-123"
        assert "lease_id" not in event.metadata


@pytest.mark.asyncio
async def test_recovery_propagates_execution_ownership_loss_without_marking_failed():
    repository = RecordingRepository()
    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        )
    )

    observer = RecordingObserver()

    class OwnershipLossRuntime:
        async def resume(
            self,
            agent_name,
            request,
            checkpoint,
            *,
            run_id=None,
            lease_id=None,
            execution_ownership_lost=None,
        ):
            raise AgentExecutionOwnershipLostError("Agent execution lost durable run ownership.")

    service = AgentRunRecoveryService(
        runtime=OwnershipLossRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
    )

    with pytest.raises(
        AgentExecutionOwnershipLostError,
        match="lost durable run ownership",
    ):
        await service.recover("run-123")

    recovered = repository.get("run-123")

    assert recovered is not None
    assert recovered.status is AgentRunStatus.RUNNING
    assert repository.fail_call is None
    assert repository.complete_call is None

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_RECOVERY_STARTED,
    ]


@pytest.mark.asyncio
async def test_recovery_emits_started_and_failed_events_on_resume_failure():
    repository = InMemoryAgentRunRepository()
    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        )
    )

    observer = RecordingObserver()

    class FailingRuntime:
        async def resume(
            self,
            agent_name,
            request,
            checkpoint,
            *,
            run_id=None,
            lease_id=None,
            execution_ownership_lost=None,
        ):
            raise ValueError("LLM provider unavailable")

    service = AgentRunRecoveryService(
        runtime=FailingRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
    )

    with pytest.raises(ValueError, match="LLM provider unavailable"):
        await service.recover("run-123")

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_RECOVERY_STARTED,
        AgentExecutionEventType.AGENT_RECOVERY_FAILED,
    ]

    assert observer.events[0].metadata == {
        "recovery_type": "failed_run",
        "recovery_attempt": 1,
    }
    assert observer.events[1].metadata["recovery_type"] == "failed_run"
    assert observer.events[1].metadata["recovery_attempt"] == 1
    assert isinstance(observer.events[1].metadata["duration_ms"], (int, float))
    assert observer.events[1].metadata["duration_ms"] >= 0
    assert observer.events[1].metadata["error_type"] == "ValueError"

    assert "lease_id" not in observer.events[1].metadata


@pytest.mark.asyncio
async def test_recovery_observer_failure_does_not_break_recovery():
    repository = InMemoryAgentRunRepository()
    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        )
    )

    service = AgentRunRecoveryService(
        runtime=FakeRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=FailingObserver(),
    )

    result = await service.recover("run-123")

    assert result.run_id == "run-123"

    run = repository.get("run-123")
    assert run is not None
    assert run.status is AgentRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_recovery_propagates_lease_ownership_loss_before_completion():
    observer = RecordingObserver()

    class OwnershipLostRepository(RecordingRepository):
        def complete_if_owner(
            self,
            run_id,
            *,
            lease_id,
            completed_at,
            output,
        ):
            self.complete_call = {
                "run_id": run_id,
                "lease_id": lease_id,
                "completed_at": completed_at,
                "output": output,
            }
            return None

    ownership_lost_repository = OwnershipLostRepository()
    ownership_lost_repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        )
    )

    service = AgentRunRecoveryService(
        runtime=FakeRuntime(),
        repository=ownership_lost_repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
    )

    with pytest.raises(
        AgentExecutionOwnershipLostError,
        match="lost lease ownership before completion",
    ):
        await service.recover("run-123")

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_RECOVERY_STARTED,
    ]

    assert ownership_lost_repository.complete_call is not None
    assert ownership_lost_repository.fail_call is None

    run = ownership_lost_repository.get("run-123")

    assert run is not None
    assert run.status is AgentRunStatus.RUNNING
    assert run.lease_id is not None


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
    repository = RecordingRepository()

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
            lease_id=None,
            execution_ownership_lost=None,
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

    assert repository.recovery_claim is not None
    assert repository.recovery_claim["lease_id"] is not None
    assert repository.fail_call is not None
    assert repository.fail_call["lease_id"] == repository.recovery_claim["lease_id"]


@pytest.mark.asyncio
async def test_final_recovery_attempt_failure_exhausts_recovery_budget():
    repository = RecordingRepository()
    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
            recovery_attempts=2,
        )
    )

    observer = RecordingObserver()

    class FailingRuntime:
        async def resume(
            self,
            agent_name,
            request,
            checkpoint,
            *,
            run_id=None,
            lease_id=None,
            execution_ownership_lost=None,
        ):
            raise ValueError("LLM provider unavailable")

    service = AgentRunRecoveryService(
        runtime=FailingRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
        max_recovery_attempts=3,
    )

    with pytest.raises(
        RecoveryExhaustedError,
        match="exhausted the maximum of 3 recovery attempts",
    ):
        await service.recover("run-123")

    assert repository.recovery_claim is not None
    assert repository.recovery_claim["max_recovery_attempts"] == 3

    run = repository.get("run-123")

    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.recovery_attempts == 3
    assert run.error_type == "RecoveryExhaustedError"
    assert run.error_message == (
        "Agent run 'run-123' exhausted the maximum of "
        "3 recovery attempts: LLM provider unavailable"
    )
    assert run.lease_id is None
    assert run.lease_expires_at is None

    assert repository.fail_call is not None
    assert repository.fail_call["error_type"] == "RecoveryExhaustedError"
    assert repository.fail_call["lease_id"] == repository.recovery_claim["lease_id"]

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_RECOVERY_STARTED,
        AgentExecutionEventType.AGENT_RECOVERY_FAILED,
    ]

    assert observer.events[0].metadata == {
        "recovery_type": "failed_run",
        "recovery_attempt": 3,
    }
    assert observer.events[1].metadata["recovery_type"] == "failed_run"
    assert observer.events[1].metadata["recovery_attempt"] == 3
    assert isinstance(observer.events[1].metadata["duration_ms"], (int, float))
    assert observer.events[1].metadata["duration_ms"] >= 0
    assert observer.events[1].metadata["error_type"] == "RecoveryExhaustedError"
    assert observer.events[1].metadata["reason"] == "max_recovery_attempts_exceeded"


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


@pytest.mark.asyncio
async def test_recovery_does_not_claim_cancellation_requested_failed_run():
    repository = InMemoryAgentRunRepository()

    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
        ).model_copy(
            update={
                "cancellation_requested": True,
                "cancellation_requested_at": datetime(
                    2026,
                    9,
                    19,
                    11,
                    59,
                    tzinfo=UTC,
                ),
            }
        )
    )

    runtime = FakeRuntime()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
    )

    with pytest.raises(
        RuntimeError,
        match="could not be claimed for recovery",
    ):
        await service.recover("run-123")

    assert runtime.calls == []

    run = repository.get("run-123")
    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.cancellation_requested is True


def stale_running_run(
    *,
    run_id: str = "stale-run-123",
    request_snapshot: AgentRunRequestSnapshot | None = None,
    recovery_attempts: int = 0,
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name="recoverable-agent",
        session_id="session-123",
        user_id="user-123",
        status=AgentRunStatus.RUNNING,
        started_at=datetime(2026, 9, 19, 11, 0, tzinfo=UTC),
        lease_id="expired-lease",
        lease_expires_at=datetime(2026, 9, 19, 11, 1, tzinfo=UTC),
        metadata={"request_id": "req-123"},
        request_snapshot=request_snapshot,
        recovery_attempts=recovery_attempts,
    )


@pytest.mark.asyncio
async def test_recover_stale_runs_limit_applies_to_stale_candidates():
    repository = InMemoryAgentRunRepository()

    repository.create(
        stale_running_run(
            run_id="stale-run",
            request_snapshot=request_snapshot(),
        )
    )
    repository.create(
        stale_running_run(
            run_id="unexpired-run",
            request_snapshot=request_snapshot(),
        ).model_copy(
            update={
                "lease_expires_at": datetime(
                    2026,
                    9,
                    19,
                    12,
                    5,
                    tzinfo=UTC,
                ),
            }
        )
    )

    runtime = FakeRuntime()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
        limit=1,
    )

    assert len(results.recovered) == 1
    assert results.recovered[0].run_id == "stale-run"


@pytest.mark.asyncio
async def test_recover_stale_runs_marks_exhausted_run_failed_without_resuming():
    repository = RecordingRepository()
    repository.create(
        stale_running_run(
            request_snapshot=request_snapshot(),
            recovery_attempts=3,
        )
    )

    runtime = FakeRuntime()
    observer = RecordingObserver()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
        max_recovery_attempts=3,
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert results.recovered == ()
    assert results.failed_run_ids == ()
    assert runtime.calls == []
    assert repository.stale_recovery_claim is None

    run = repository.get("stale-run-123")

    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.recovery_attempts == 3
    assert run.error_type == "RecoveryExhaustedError"
    assert run.error_message == (
        "Agent run 'stale-run-123' exhausted the maximum of " "3 recovery attempts."
    )
    assert run.lease_id is None
    assert run.lease_expires_at is None

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_RECOVERY_FAILED,
    ]

    assert observer.events[0].metadata == {
        "recovery_type": "stale_run",
        "recovery_attempt": 3,
        "error_type": "RecoveryExhaustedError",
        "reason": "max_recovery_attempts_exceeded",
    }


@pytest.mark.asyncio
async def test_recover_stale_runs_does_not_emit_duplicate_exhaustion_event():
    repository = RecordingRepository()
    repository.create(
        stale_running_run(
            request_snapshot=request_snapshot(),
            recovery_attempts=3,
        )
    )

    observer = RecordingObserver()

    service = AgentRunRecoveryService(
        runtime=FakeRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
        max_recovery_attempts=3,
    )

    stale_before = datetime(2026, 9, 19, 12, 0, tzinfo=UTC)

    first_results = await service.recover_stale_runs(
        stale_before=stale_before,
    )
    second_results = await service.recover_stale_runs(
        stale_before=stale_before,
    )

    assert first_results.recovered == ()
    assert first_results.failed_run_ids == ()
    assert second_results.recovered == ()
    assert second_results.failed_run_ids == ()

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_RECOVERY_FAILED,
    ]

    run = repository.get("stale-run-123")

    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.recovery_attempts == 3


@pytest.mark.asyncio
async def test_recover_stale_runs_claims_and_resumes_expired_run():
    repository = RecordingRepository()

    repository.create(
        stale_running_run(
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

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert len(results.recovered) == 1
    assert results.recovered[0].run_id == "stale-run-123"
    assert results.recovered[0].response.output == "Recovered successfully."

    recovered = repository.get("stale-run-123")

    assert recovered is not None
    assert recovered.status is AgentRunStatus.COMPLETED
    assert recovered.output == "Recovered successfully."
    assert recovered.lease_id is None
    assert recovered.lease_expires_at is None

    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_recover_stale_runs_emits_started_and_completed_events():
    repository = InMemoryAgentRunRepository()
    repository.create(
        stale_running_run(
            request_snapshot=request_snapshot(),
        )
    )

    observer = RecordingObserver()

    service = AgentRunRecoveryService(
        runtime=FakeRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert len(results.recovered) == 1

    assert [event.event_type for event in observer.events] == [
        AgentExecutionEventType.AGENT_RECOVERY_STARTED,
        AgentExecutionEventType.AGENT_RECOVERY_COMPLETED,
    ]

    assert observer.events[0].metadata == {
        "recovery_type": "stale_run",
        "recovery_attempt": 1,
    }
    assert observer.events[1].metadata["recovery_type"] == "stale_run"
    assert observer.events[1].metadata["recovery_attempt"] == 1
    assert isinstance(observer.events[1].metadata["duration_ms"], (int, float))
    assert observer.events[1].metadata["duration_ms"] >= 0

    for event in observer.events:
        assert event.run_id == "stale-run-123"
        assert event.agent_name == "recoverable-agent"
        assert event.session_id == "session-123"
        assert event.user_id == "user-123"
        assert "lease_id" not in event.metadata


@pytest.mark.asyncio
async def test_recover_stale_runs_ignores_unexpired_run():
    repository = InMemoryAgentRunRepository()

    repository.create(
        stale_running_run(
            request_snapshot=request_snapshot(),
        ).model_copy(
            update={
                "lease_expires_at": datetime(
                    2026,
                    9,
                    19,
                    12,
                    5,
                    tzinfo=UTC,
                ),
            }
        )
    )

    runtime = FakeRuntime()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert results.recovered == ()
    assert results.failed_run_ids == ()
    assert runtime.calls == []

    run = repository.get("stale-run-123")

    assert run is not None
    assert run.status is AgentRunStatus.RUNNING
    assert run.lease_id == "expired-lease"


@pytest.mark.asyncio
async def test_recover_stale_runs_ignores_run_without_lease_expiry():
    repository = InMemoryAgentRunRepository()

    repository.create(
        stale_running_run(
            request_snapshot=request_snapshot(),
        ).model_copy(
            update={
                "lease_expires_at": None,
            }
        )
    )

    runtime = FakeRuntime()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert results.recovered == ()
    assert results.failed_run_ids == ()
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_recover_stale_runs_ignores_cancellation_requested_run():
    repository = InMemoryAgentRunRepository()

    repository.create(
        stale_running_run(
            request_snapshot=request_snapshot(),
        ).model_copy(
            update={
                "cancellation_requested": True,
                "cancellation_requested_at": datetime(
                    2026,
                    9,
                    19,
                    11,
                    59,
                    tzinfo=UTC,
                ),
            }
        )
    )

    runtime = FakeRuntime()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert results.recovered == ()
    assert results.failed_run_ids == ()
    assert runtime.calls == []

    run = repository.get("stale-run-123")

    assert run is not None
    assert run.status is AgentRunStatus.RUNNING
    assert run.cancellation_requested is True
    assert run.lease_id == "expired-lease"


@pytest.mark.asyncio
async def test_recover_stale_runs_marks_run_failed_when_checkpoint_is_missing():
    repository = RecordingRepository()

    repository.create(
        stale_running_run(
            request_snapshot=request_snapshot(),
        )
    )

    service = AgentRunRecoveryService(
        runtime=FakeRuntime(),
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(None),
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert results.recovered == ()
    assert results.failed_run_ids == ("stale-run-123",)

    run = repository.get("stale-run-123")

    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.error_type == "RuntimeError"
    assert "has no execution checkpoint" in run.error_message
    assert run.lease_id is None
    assert run.lease_expires_at is None


@pytest.mark.asyncio
async def test_recover_stale_runs_continues_after_one_run_fails():
    repository = InMemoryAgentRunRepository()

    repository.create(
        stale_running_run(
            run_id="stale-failing",
            request_snapshot=request_snapshot(),
        )
    )
    repository.create(
        stale_running_run(
            run_id="stale-success",
            request_snapshot=request_snapshot(),
        )
    )

    class SelectiveRuntime(FakeRuntime):
        async def resume(
            self,
            agent_name,
            request,
            checkpoint,
            *,
            run_id=None,
            lease_id=None,
            execution_ownership_lost=None,
        ):
            self.calls.append(
                {
                    "agent_name": agent_name,
                    "request": request,
                    "checkpoint": checkpoint,
                    "run_id": run_id,
                    "execution_ownership_lost": execution_ownership_lost,
                }
            )

            if run_id == "stale-failing":
                raise ValueError("temporary provider failure")

            return AgentResponse(
                agent_name=agent_name,
                output="Recovered successfully.",
                session_id=request.session_id,
            )

    runtime = SelectiveRuntime()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert len(results.recovered) == 1
    assert results.recovered[0].run_id == "stale-success"
    assert results.failed_run_ids == ("stale-failing",)

    failed = repository.get("stale-failing")
    succeeded = repository.get("stale-success")

    assert failed is not None
    assert failed.status is AgentRunStatus.FAILED
    assert failed.error_type == "ValueError"
    assert failed.error_message == "temporary provider failure"

    assert succeeded is not None
    assert succeeded.status is AgentRunStatus.COMPLETED
    assert succeeded.output == "Recovered successfully."

    assert {call["run_id"] for call in runtime.calls} == {
        "stale-failing",
        "stale-success",
    }


@pytest.mark.asyncio
async def test_recover_stale_runs_resumes_with_new_lease_ownership():
    repository = RecordingRepository()

    repository.create(
        stale_running_run(
            request_snapshot=request_snapshot(),
        )
    )

    original_lease_id = "expired-lease"
    runtime = FakeRuntime()
    checkpoint_repository = FakeCheckpointRepository(checkpoint())

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=checkpoint_repository,
        lease_seconds=60,
    )

    results = await service.recover_stale_runs(
        stale_before=datetime(2026, 9, 19, 12, 0, tzinfo=UTC),
    )

    assert len(results.recovered) == 1

    assert repository.stale_recovery_claim is not None
    new_lease_id = repository.stale_recovery_claim["lease_id"]

    assert new_lease_id is not None
    assert new_lease_id != original_lease_id

    assert repository.complete_call is not None
    assert repository.complete_call["lease_id"] == new_lease_id

    recovered = repository.get("stale-run-123")

    assert recovered is not None
    assert recovered.status is AgentRunStatus.COMPLETED
    assert recovered.lease_id is None
    assert recovered.lease_expires_at is None


@pytest.mark.asyncio
async def test_recover_rejects_already_exhausted_run():
    repository = RecordingRepository()
    repository.create(
        failed_run(
            request_snapshot=request_snapshot(),
            recovery_attempts=3,
        )
    )

    runtime = FakeRuntime()
    observer = RecordingObserver()

    service = AgentRunRecoveryService(
        runtime=runtime,
        repository=repository,
        checkpoints_repository=FakeCheckpointRepository(checkpoint()),
        observer=observer,
        max_recovery_attempts=3,
    )

    with pytest.raises(
        RecoveryExhaustedError,
        match="exhausted the maximum of 3 recovery attempts",
    ):
        await service.recover("run-123")

    run = repository.get("run-123")

    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.recovery_attempts == 3
    assert run.error_type == "RuntimeError"
    assert run.error_message == "original failure"
    assert run.lease_id is None
    assert runtime.calls == []
    assert repository.recovery_claim is None
    assert observer.events == []


def test_recovery_service_rejects_non_positive_max_recovery_attempts():
    with pytest.raises(
        ValueError,
        match="max_recovery_attempts must be greater than zero",
    ):
        AgentRunRecoveryService(
            runtime=FakeRuntime(),
            repository=RecordingRepository(),
            checkpoints_repository=FakeCheckpointRepository(checkpoint()),
            max_recovery_attempts=0,
        )
