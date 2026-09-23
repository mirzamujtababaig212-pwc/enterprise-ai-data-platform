from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_platform.agents.exceptions import AgentExecutionOwnershipLostError
from ai_platform.agents.models import AgentRequest, AgentResponse
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)

from app.control_plane.agent_run_events.repository import (
    AgentRunEventsRepository,
)
from app.control_plane.agent_runs.admission import AgentRunAdmissionResult
from app.control_plane.agent_runs.exceptions import (
    AgentRunAccessDeniedError,
    AgentRunAdmissionRejectedError,
    AgentRunIdempotencyConflictError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_runs.lease import heartbeat_loop
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_steps.repository import (
    AgentRunStepsRepository,
)


def _response(
    *,
    agent_name: str = "enterprise-analyst",
    output="completed",
    session_id: str | None = "session-1",
) -> AgentResponse:
    return AgentResponse(
        agent_name=agent_name,
        output=output,
        session_id=session_id,
        metadata={"source": "test"},
    )


def _repository() -> Mock:
    repository = Mock(spec=AgentRunRepository)
    repository.create.side_effect = lambda run: run
    repository.update.side_effect = lambda run: run

    def complete_if_owner(
        run_id: str,
        *,
        lease_id: str,
        completed_at,
        output,
    ) -> AgentRun:
        running = repository.update.call_args_list[0].args[0]
        completed = running.transition_to(
            AgentRunStatus.COMPLETED,
        ).model_copy(
            update={
                "completed_at": completed_at,
                "output": output,
                "lease_id": None,
                "lease_expires_at": None,
            }
        )
        repository.completed_run = completed
        return completed

    def fail_if_owner(
        run_id: str,
        *,
        lease_id: str,
        completed_at,
        error_type: str,
        error_message: str,
    ) -> AgentRun:
        running = repository.update.call_args_list[0].args[0]
        failed = running.transition_to(
            AgentRunStatus.FAILED,
        ).model_copy(
            update={
                "completed_at": completed_at,
                "error_type": error_type,
                "error_message": error_message,
                "lease_id": None,
                "lease_expires_at": None,
            }
        )
        repository.failed_run = failed
        return failed

    def cancel_if_owner(
        run_id: str,
        *,
        lease_id: str,
        completed_at,
    ) -> AgentRun:
        running = repository.update.call_args_list[0].args[0]
        cancelled = running.transition_to(
            AgentRunStatus.CANCELLED,
        ).model_copy(
            update={
                "completed_at": completed_at,
                "lease_id": None,
                "lease_expires_at": None,
            }
        )
        repository.cancelled_run = cancelled
        return cancelled

    repository.complete_if_owner.side_effect = complete_if_owner
    repository.fail_if_owner.side_effect = fail_if_owner
    repository.cancel_if_owner.side_effect = cancel_if_owner

    return repository


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(self, event: AgentExecutionEvent) -> None:
        self.events.append(event)


class FailingObserver:
    def __init__(self) -> None:
        self.calls = 0

    async def record(self, event: AgentExecutionEvent) -> None:
        self.calls += 1
        raise RuntimeError("event persistence unavailable")


@pytest.mark.asyncio
async def test_execute_persists_authenticated_tenant_in_run_and_snapshot() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-tenant-1",
        user_id="user-tenant-1",
        principal="api_key:tenant-principal",
        tenant_id="tenant-acme",
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=request,
    )

    assert result.run_id

    pending = repository.create.call_args.args[0]
    running = repository.update.call_args_list[0].args[0]
    completed = repository.completed_run

    assert pending.tenant_id == "tenant-acme"
    assert pending.request_snapshot.tenant_id == "tenant-acme"

    assert running.tenant_id == "tenant-acme"
    assert running.request_snapshot.tenant_id == "tenant-acme"

    assert completed.tenant_id == "tenant-acme"
    assert completed.request_snapshot.tenant_id == "tenant-acme"


@pytest.mark.asyncio
async def test_execute_persists_pending_running_and_completed_lifecycle() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    response = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
        ),
    )

    assert isinstance(response, AgentRunExecutionResult)
    assert response.run_id
    assert response.response.output == "completed"

    assert repository.create.call_count == 1
    assert repository.update.call_count == 1
    repository.complete_if_owner.assert_called_once()

    pending = repository.create.call_args.args[0]
    running = repository.update.call_args_list[0].args[0]
    completed = repository.completed_run

    assert pending.status == AgentRunStatus.PENDING
    assert pending.agent_name == "enterprise-analyst"
    assert pending.session_id == "session-1"
    assert pending.user_id == "user-1"
    assert pending.started_at is None
    assert pending.completed_at is None

    assert running.run_id == pending.run_id
    assert running.lease_id is not None
    assert running.lease_expires_at is not None
    assert running.lease_expires_at > running.started_at

    assert repository.complete_if_owner.call_args.args[0] == running.run_id
    assert repository.complete_if_owner.call_args.kwargs["lease_id"] == running.lease_id
    assert running.status == AgentRunStatus.RUNNING
    assert running.started_at is not None
    assert running.completed_at is None

    assert completed.run_id == pending.run_id
    assert completed.status == AgentRunStatus.COMPLETED
    assert completed.started_at == running.started_at
    assert completed.completed_at is not None
    assert completed.output == "completed"
    assert completed.error_type is None
    assert completed.error_message is None
    assert completed.lease_id is None
    assert completed.lease_expires_at is None

    runtime.run.assert_awaited_once_with(
        "enterprise-analyst",
        AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
        ),
        lease_id=running.lease_id,
        run_id=pending.run_id,
        execution_ownership_lost=runtime.run.await_args.kwargs["execution_ownership_lost"],
    )


@pytest.mark.asyncio
async def test_execute_cancels_when_cancellation_is_requested_during_registration_gap():
    repository = _repository()

    class BlockingRuntime:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False

        async def run(
            self,
            agent_name,
            request,
            *,
            lease_id=None,
            run_id=None,
            execution_ownership_lost=None,
        ):
            self.started.set()

            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    runtime = BlockingRuntime()

    running_run = None
    cancellation_checked = False

    original_update = repository.update.side_effect

    def capture_running_run(run):
        nonlocal running_run
        running_run = run
        return original_update(run)

    repository.update.side_effect = capture_running_run

    def get_with_registration_race(run_id: str):
        nonlocal cancellation_checked

        if running_run is None:
            return None

        if not cancellation_checked:
            cancellation_checked = True
            return running_run.model_copy(
                update={
                    "cancellation_requested": True,
                    "cancellation_requested_at": datetime.now(UTC),
                }
            )

        return running_run

    repository.get.side_effect = get_with_registration_race

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    execution_task = asyncio.create_task(
        service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
            ),
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    assert cancellation_checked is True
    assert runtime.cancelled is True
    repository.cancel_if_owner.assert_called_once()
    assert hasattr(repository, "cancelled_run")
    assert repository.cancelled_run.status is AgentRunStatus.CANCELLED
    repository.complete_if_owner.assert_not_called()
    repository.fail_if_owner.assert_not_called()


@pytest.mark.asyncio
async def test_execute_uses_one_run_id_across_lifecycle() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(input="Hello"),
    )

    created_run = repository.create.call_args.args[0]
    running_run = repository.update.call_args_list[0].args[0]
    completed_run_id = repository.complete_if_owner.call_args.args[0]

    assert created_run.run_id == running_run.run_id
    assert completed_run_id == running_run.run_id


@pytest.mark.asyncio
async def test_execute_propagates_execution_ownership_loss_without_terminal_persistence():
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(
        side_effect=AgentExecutionOwnershipLostError("Agent execution lost durable run ownership."),
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentExecutionOwnershipLostError,
        match="lost durable run ownership",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Ownership loss"),
        )

    assert repository.create.call_count == 1
    assert repository.update.call_count == 1
    repository.fail_if_owner.assert_not_called()
    repository.cancel_if_owner.assert_not_called()
    repository.complete_if_owner.assert_not_called()


@pytest.mark.asyncio
async def test_execute_preserves_agent_response_without_rebuilding_it() -> None:
    repository = _repository()

    response = _response(
        output={"answer": "structured"},
        session_id="session-42",
    )

    runtime = Mock()
    runtime.run = AsyncMock(return_value=response)

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Return structured output",
            session_id="session-42",
        ),
    )

    assert isinstance(result, AgentRunExecutionResult)
    assert result.response is response
    assert result.run_id


@pytest.mark.asyncio
async def test_execute_persists_failed_run_and_reraises_runtime_error() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(
        side_effect=RuntimeError("agent execution failed"),
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="agent execution failed"):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Fail"),
        )

    assert repository.create.call_count == 1
    assert repository.update.call_count == 1
    repository.fail_if_owner.assert_called_once()

    pending = repository.create.call_args.args[0]
    running = repository.update.call_args_list[0].args[0]
    failed = repository.failed_run

    assert pending.status == AgentRunStatus.PENDING
    assert running.status == AgentRunStatus.RUNNING
    assert running.lease_id is not None
    assert running.lease_expires_at is not None

    assert repository.fail_if_owner.call_args.args[0] == running.run_id
    assert repository.fail_if_owner.call_args.kwargs["lease_id"] == running.lease_id

    assert failed.run_id == pending.run_id
    assert failed.status == AgentRunStatus.FAILED
    assert failed.started_at == running.started_at
    assert failed.completed_at is not None
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "agent execution failed"
    assert failed.output is None
    assert failed.lease_id is None
    assert failed.lease_expires_at is None


@pytest.mark.asyncio
async def test_execute_persists_failed_run_for_lookup_error() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(
        side_effect=LookupError("agent not found"),
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(LookupError, match="agent not found"):
        await service.execute(
            agent_name="missing-agent",
            request=AgentRequest(input="Hello"),
        )

    repository.fail_if_owner.assert_called_once()

    failed = repository.failed_run
    running = repository.update.call_args_list[0].args[0]

    assert failed.status == AgentRunStatus.FAILED
    assert failed.error_type == "LookupError"
    assert failed.error_message == "agent not found"
    assert repository.fail_if_owner.call_args.args[0] == running.run_id
    assert repository.fail_if_owner.call_args.kwargs["lease_id"] == running.lease_id


@pytest.mark.asyncio
async def test_execute_does_not_mark_failed_when_running_persistence_fails() -> None:
    repository = Mock(spec=AgentRunRepository)

    pending = Mock(spec=AgentRun)
    repository.create.return_value = pending
    repository.update.side_effect = RuntimeError("running persistence failed")

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="running persistence failed"):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Hello"),
        )

    runtime.run.assert_not_awaited()
    repository.create.assert_called_once()
    repository.update.assert_called_once()


@pytest.mark.asyncio
async def test_execute_does_not_attempt_failed_update_when_runtime_never_starts() -> None:
    repository = Mock(spec=AgentRunRepository)
    repository.create.side_effect = RuntimeError("initial persistence failed")

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="initial persistence failed"):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Hello"),
        )

    repository.update.assert_not_called()
    runtime.run.assert_not_awaited()


def test_get_run_returns_repository_result() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
        output="completed",
    )
    repository.get.return_value = run

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = service.get_run("run-123")

    assert result is run
    repository.get.assert_called_once_with("run-123")


@pytest.mark.asyncio
async def test_execute_preserves_runtime_error_when_failed_persistence_fails() -> None:
    repository = _repository()

    original_error = RuntimeError("agent execution failed")
    runtime = Mock()
    runtime.run = AsyncMock(side_effect=original_error)

    repository.fail_if_owner.side_effect = RuntimeError("failed-state persistence failed")

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="agent execution failed") as exc_info:
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Fail"),
        )

    assert exc_info.value is original_error
    assert repository.create.call_count == 1
    assert repository.update.call_count == 1
    repository.fail_if_owner.assert_called_once()


@pytest.mark.asyncio
async def test_execute_surfaces_completed_persistence_failure() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    repository.complete_if_owner.side_effect = RuntimeError("completed-state persistence failed")

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="completed-state persistence failed",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Complete"),
        )

    runtime.run.assert_awaited_once()
    assert repository.create.call_count == 1
    assert repository.update.call_count == 1
    repository.complete_if_owner.assert_called_once()


def test_get_run_returns_none_for_missing_run() -> None:
    repository = _repository()
    repository.get.return_value = None

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = service.get_run("missing-run")

    assert result is None
    repository.get.assert_called_once_with("missing-run")


def test_get_run_rejects_partial_identity_context() -> None:
    repository = _repository()
    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    with pytest.raises(
        AgentRunAccessDeniedError,
        match="Both tenant_id and principal are required",
    ):
        service.get_run(
            "run-123",
            tenant_id="tenant-acme",
        )

    repository.get.assert_not_called()
    repository.get_for_tenant.assert_not_called()


def test_cancel_rejects_partial_identity_context() -> None:
    repository = _repository()
    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    with pytest.raises(
        AgentRunAccessDeniedError,
        match="Both tenant_id and principal are required",
    ):
        service.cancel(
            "run-123",
            principal="api_key:test-owner",
        )

    repository.get.assert_not_called()
    repository.get_for_tenant.assert_not_called()


def test_list_events_rejects_partial_identity_context() -> None:
    repository = _repository()
    events_repository = Mock(spec=AgentRunEventsRepository)

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        events_repository=events_repository,
    )

    with pytest.raises(
        AgentRunAccessDeniedError,
        match="Both tenant_id and principal are required",
    ):
        service.list_events(
            "run-123",
            tenant_id="tenant-acme",
        )

    repository.get.assert_not_called()
    repository.get_for_tenant.assert_not_called()
    events_repository.list.assert_not_called()


def test_list_runs_delegates_filters_and_limit() -> None:
    repository = _repository()

    runs = [
        AgentRun(
            run_id="run-1",
            agent_name="enterprise-analyst",
            status=AgentRunStatus.COMPLETED,
        ),
        AgentRun(
            run_id="run-2",
            agent_name="enterprise-analyst",
            status=AgentRunStatus.FAILED,
        ),
    ]
    repository.list.return_value = runs

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = service.list_runs(
        agent_name="enterprise-analyst",
        session_id="session-1",
        user_id="user-1",
        status=AgentRunStatus.COMPLETED,
        limit=25,
    )

    assert result == runs
    repository.list.assert_called_once_with(
        tenant_id=None,
        agent_name="enterprise-analyst",
        session_id="session-1",
        user_id="user-1",
        status=AgentRunStatus.COMPLETED,
        limit=25,
    )


def test_list_events_delegates_to_event_repository() -> None:
    repository = _repository()
    events_repository = Mock(spec=AgentRunEventsRepository)

    run = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )
    repository.get.return_value = run

    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="enterprise-analyst",
            run_id="run-123",
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_COMPLETED,
            agent_name="enterprise-analyst",
            run_id="run-123",
        ),
    ]
    events_repository.list.return_value = events

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        events_repository=events_repository,
    )

    result = service.list_events(
        "run-123",
        limit=25,
    )

    assert result == events
    repository.get.assert_called_once_with("run-123")
    events_repository.list.assert_called_once_with(
        "run-123",
        limit=25,
    )


def test_list_events_raises_for_missing_run() -> None:
    repository = _repository()
    repository.get.return_value = None

    events_repository = Mock(spec=AgentRunEventsRepository)
    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        events_repository=events_repository,
    )

    with pytest.raises(
        LookupError,
        match="Agent run 'missing-run' was not found.",
    ):
        service.list_events("missing-run")

    repository.get.assert_called_once_with("missing-run")
    events_repository.list.assert_not_called()


def test_list_events_requires_event_repository() -> None:
    repository = _repository()
    repository.get.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent run events repository is not configured.",
    ):
        service.list_events("run-123")


@pytest.mark.asyncio
async def test_execute_rejects_run_before_runtime_when_admission_denied() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    admission_policy = Mock()
    admission_policy.evaluate = AsyncMock(
        return_value=AgentRunAdmissionResult(
            allowed=False,
            reason="agent is not approved for this environment",
        )
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        admission_policy=admission_policy,
    )

    with pytest.raises(
        AgentRunAdmissionRejectedError,
        match="agent is not approved for this environment",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
            ),
        )

    assert repository.create.call_count == 1
    assert repository.update.call_count == 1
    runtime.run.assert_not_awaited()

    pending = repository.create.call_args.args[0]
    rejected = repository.update.call_args.args[0]

    assert pending.status == AgentRunStatus.PENDING
    assert pending.started_at is None
    assert pending.completed_at is None

    assert rejected.run_id == pending.run_id
    assert rejected.status == AgentRunStatus.REJECTED
    assert rejected.started_at is None
    assert rejected.completed_at is not None
    assert rejected.metadata["admission"] == {
        "allowed": False,
        "reason": "agent is not approved for this environment",
    }

    admission_policy.evaluate.assert_awaited_once()
    call = admission_policy.evaluate.await_args

    assert call.kwargs["agent_name"] == "enterprise-analyst"
    assert call.kwargs["request"].user_id == "user-1"
    assert call.kwargs["run"].run_id == pending.run_id
    assert call.kwargs["run"].status == AgentRunStatus.PENDING


@pytest.mark.asyncio
async def test_execute_allows_run_when_admission_policy_allows() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    admission_policy = Mock()
    admission_policy.evaluate = AsyncMock(
        return_value=AgentRunAdmissionResult(
            allowed=True,
        )
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        admission_policy=admission_policy,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(input="Explain the platform"),
    )

    assert result.response.output == "completed"
    runtime.run.assert_awaited_once()

    assert repository.create.call_count == 1
    assert repository.update.call_count == 1
    repository.complete_if_owner.assert_called_once()

    running = repository.update.call_args_list[0].args[0]
    completed = repository.completed_run

    assert running.status == AgentRunStatus.RUNNING
    assert running.lease_id is not None
    assert running.lease_expires_at is not None
    assert running.lease_expires_at > running.started_at

    assert repository.complete_if_owner.call_args.args[0] == running.run_id
    assert repository.complete_if_owner.call_args.kwargs["lease_id"] == running.lease_id

    assert completed.status == AgentRunStatus.COMPLETED
    assert completed.lease_id is None
    assert completed.lease_expires_at is None


@pytest.mark.asyncio
async def test_heartbeat_loop_renews_owned_running_run() -> None:
    repository = Mock(spec=AgentRunRepository)

    renewed_run = AgentRun(
        run_id="run-heartbeat",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    repository.heartbeat.return_value = renewed_run

    sleep_calls = 0

    async def sleep_once(seconds: float) -> None:
        nonlocal sleep_calls

        assert seconds == 20
        sleep_calls += 1

        if sleep_calls == 2:
            raise asyncio.CancelledError

    with patch(
        "app.control_plane.agent_runs.lease.asyncio.sleep",
        side_effect=sleep_once,
    ):
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_loop(
                repository,
                run_id="run-heartbeat",
                lease_id="lease-123",
                lease_seconds=60,
            )

    repository.heartbeat.assert_called_once()

    call = repository.heartbeat.call_args
    assert call.args[0] == "run-heartbeat"
    assert call.kwargs["lease_id"] == "lease-123"
    assert call.kwargs["lease_expires_at"] > datetime.now(UTC)

    assert sleep_calls == 2


@pytest.mark.asyncio
async def test_heartbeat_loop_stops_when_lease_ownership_is_lost() -> None:
    repository = Mock(spec=AgentRunRepository)
    repository.heartbeat.return_value = None

    async def sleep_once(seconds: float) -> None:
        assert seconds == 20

    with patch(
        "app.control_plane.agent_runs.lease.asyncio.sleep",
        side_effect=sleep_once,
    ):
        await heartbeat_loop(
            repository,
            run_id="run-heartbeat",
            lease_id="lease-123",
            lease_seconds=60,
        )

    repository.heartbeat.assert_called_once()


@pytest.mark.asyncio
async def test_heartbeat_loop_survives_heartbeat_persistence_failure() -> None:
    repository = Mock(spec=AgentRunRepository)

    renewed_run = AgentRun(
        run_id="run-heartbeat",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    repository.heartbeat.side_effect = [RuntimeError("database unavailable"), renewed_run]

    sleep_calls = 0

    async def sleep_once(seconds: float) -> None:
        nonlocal sleep_calls

        assert seconds == 20
        sleep_calls += 1

        if sleep_calls == 3:
            raise asyncio.CancelledError

    with patch(
        "app.control_plane.agent_runs.lease.asyncio.sleep",
        side_effect=sleep_once,
    ):
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_loop(
                repository,
                run_id="run-heartbeat",
                lease_id="lease-123",
                lease_seconds=60,
            )

    assert repository.heartbeat.call_count == 2
    assert sleep_calls == 3


@pytest.mark.asyncio
async def test_execute_starts_and_cancels_heartbeat_task() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        lease_seconds=60,
    )

    original_create_task = asyncio.create_task
    created_tasks = []

    def create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    with patch(
        "app.control_plane.agent_runs.application_service.asyncio.create_task",
        side_effect=create_task,
    ) as create_task_mock:
        result = await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
            ),
        )

    assert result.response.output == "completed"
    create_task_mock.assert_called_once()

    assert len(created_tasks) == 1
    assert created_tasks[0].done()
    assert created_tasks[0].cancelled()

    repository.complete_if_owner.assert_called_once()


@pytest.mark.asyncio
async def test_execute_persists_cancelled_run_and_emits_audit_event() -> None:
    repository = _repository()

    runtime = Mock()

    started = asyncio.Event()

    async def run_agent(*args, **kwargs):
        started.set()
        await asyncio.Future()

    runtime.run = AsyncMock(side_effect=run_agent)

    from app.control_plane.agent_runs.cancellation import (
        AgentRunCancellationRegistry,
    )

    registry = AgentRunCancellationRegistry()
    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        observer=observer,
        cancellation_registry=registry,
    )

    execution_task = asyncio.create_task(
        service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Long running task",
                session_id="session-1",
                user_id="user-1",
            ),
        )
    )

    await started.wait()

    running = repository.update.call_args_list[0].args[0]

    assert running.status == AgentRunStatus.RUNNING
    assert registry.cancel(running.run_id) is True

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    repository.cancel_if_owner.assert_called_once()

    cancelled = repository.cancelled_run

    assert cancelled.run_id == running.run_id
    assert cancelled.status == AgentRunStatus.CANCELLED
    assert cancelled.started_at == running.started_at
    assert cancelled.completed_at is not None
    assert cancelled.output is None
    assert cancelled.error_type is None
    assert cancelled.error_message is None
    assert cancelled.lease_id is None
    assert cancelled.lease_expires_at is None

    assert repository.cancel_if_owner.call_args.args[0] == running.run_id
    assert repository.cancel_if_owner.call_args.kwargs["lease_id"] == running.lease_id
    assert repository.cancel_if_owner.call_args.kwargs["completed_at"] is not None

    repository.fail_if_owner.assert_not_called()
    repository.complete_if_owner.assert_not_called()

    assert len(observer.events) == 1

    event = observer.events[0]

    assert event.event_type is AgentExecutionEventType.AGENT_CANCELLED
    assert event.agent_name == running.agent_name
    assert event.run_id == running.run_id
    assert event.session_id == running.session_id
    assert event.user_id == running.user_id
    assert event.metadata == {}
    assert event.tool_round is None
    assert event.tool_name is None
    assert event.call_id is None
    assert event.provider is None
    assert event.model is None


@pytest.mark.asyncio
async def test_execute_cancellation_does_not_emit_event_after_ownership_loss() -> None:
    repository = _repository()
    repository.cancel_if_owner.side_effect = lambda *args, **kwargs: None

    runtime = Mock()

    started = asyncio.Event()

    async def run_agent(*args, **kwargs):
        started.set()
        await asyncio.Future()

    runtime.run = AsyncMock(side_effect=run_agent)

    from app.control_plane.agent_runs.cancellation import (
        AgentRunCancellationRegistry,
    )

    registry = AgentRunCancellationRegistry()
    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        observer=observer,
        cancellation_registry=registry,
    )

    execution_task = asyncio.create_task(
        service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Long running task",
                session_id="session-1",
                user_id="user-1",
            ),
        )
    )

    await started.wait()

    running = repository.update.call_args_list[0].args[0]

    assert registry.cancel(running.run_id) is True

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    repository.cancel_if_owner.assert_called_once()
    assert observer.events == []


@pytest.mark.asyncio
async def test_execute_cancellation_continues_when_observer_fails() -> None:
    repository = _repository()

    runtime = Mock()

    started = asyncio.Event()

    async def run_agent(*args, **kwargs):
        started.set()
        await asyncio.Future()

    runtime.run = AsyncMock(side_effect=run_agent)

    from app.control_plane.agent_runs.cancellation import (
        AgentRunCancellationRegistry,
    )

    registry = AgentRunCancellationRegistry()
    observer = FailingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        observer=observer,
        cancellation_registry=registry,
    )

    execution_task = asyncio.create_task(
        service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Long running task",
                session_id="session-1",
                user_id="user-1",
            ),
        )
    )

    await started.wait()

    running = repository.update.call_args_list[0].args[0]

    assert registry.cancel(running.run_id) is True

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    repository.cancel_if_owner.assert_called_once()

    cancelled = repository.cancelled_run
    assert cancelled.status == AgentRunStatus.CANCELLED
    assert observer.calls == 1


def test_cancel_raises_lookup_error_for_missing_run() -> None:
    repository = _repository()
    repository.get.return_value = None

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        LookupError,
        match="Agent run 'missing-run' was not found.",
    ):
        service.cancel("missing-run")

    repository.get.assert_called_once_with("missing-run")


def test_cancel_rejects_non_running_run() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="completed-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )
    repository.get.return_value = run

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        ValueError,
        match="Agent run 'completed-run' is not cancellable from status 'completed'",
    ):
        service.cancel("completed-run")

    repository.get.assert_called_once_with("completed-run")
    repository.request_cancellation.assert_not_called()


def test_cancel_persists_intent_then_signals_registered_running_task() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="running-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        started_at=datetime.now(UTC),
    )
    requested_run = run.model_copy(
        update={
            "cancellation_requested": True,
            "cancellation_requested_at": datetime.now(UTC),
        }
    )
    repository.get.return_value = run
    repository.request_cancellation.return_value = requested_run

    cancellation_registry = Mock()
    cancellation_registry.cancel.return_value = True

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        cancellation_registry=cancellation_registry,
    )

    result = service.cancel("running-run")

    assert result is requested_run
    repository.get.assert_called_once_with("running-run")
    repository.request_cancellation.assert_called_once()
    request_kwargs = repository.request_cancellation.call_args.kwargs
    assert request_kwargs["requested_at"].tzinfo is UTC
    cancellation_registry.cancel.assert_called_once_with("running-run")


def test_cancel_returns_durable_intent_when_local_task_is_missing() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="running-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        started_at=datetime.now(UTC),
    )
    requested_run = run.model_copy(
        update={
            "cancellation_requested": True,
            "cancellation_requested_at": datetime.now(UTC),
        }
    )
    repository.get.return_value = run
    repository.request_cancellation.return_value = requested_run

    cancellation_registry = Mock()
    cancellation_registry.cancel.return_value = False

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        cancellation_registry=cancellation_registry,
    )

    result = service.cancel("running-run")

    assert result is requested_run
    repository.request_cancellation.assert_called_once()
    cancellation_registry.cancel.assert_called_once_with("running-run")


def test_cancel_rejects_when_cancellation_intent_cannot_be_persisted() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="running-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        started_at=datetime.now(UTC),
    )
    repository.get.return_value = run
    repository.request_cancellation.return_value = None

    runtime = Mock()
    cancellation_registry = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        cancellation_registry=cancellation_registry,
    )

    with pytest.raises(
        RuntimeError,
        match="could not accept cancellation",
    ):
        service.cancel("running-run")

    repository.request_cancellation.assert_called_once()
    cancellation_registry.cancel.assert_not_called()


def test_cancel_does_not_require_local_registry_when_intent_is_persisted() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="running-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        started_at=datetime.now(UTC),
    )
    requested_run = run.model_copy(
        update={
            "cancellation_requested": True,
            "cancellation_requested_at": datetime.now(UTC),
        }
    )
    repository.get.return_value = run
    repository.request_cancellation.return_value = requested_run

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = service.cancel("running-run")

    assert result is requested_run
    repository.request_cancellation.assert_called_once()


def _idempotent_run(
    *,
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
    run_id: str = "existing-run",
    agent_name: str = "enterprise-analyst",
    output="completed",
    user_id: str = "user-1",
    idempotency_key: str = "request-123",
) -> AgentRun:
    request = AgentRequest(
        input="Explain the platform",
        session_id="session-1",
        user_id=user_id,
        metadata={"request": "same"},
    )

    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        session_id=request.session_id,
        user_id=user_id,
        idempotency_key=idempotency_key,
        status=status,
        output=output,
        metadata={"source": "persisted"},
        request_snapshot=AgentRunRequestSnapshot.from_request(request),
        error_message=("persisted failure" if status is AgentRunStatus.FAILED else None),
    )


@pytest.mark.asyncio
async def test_execute_without_idempotency_key_creates_new_run() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            user_id="user-1",
        ),
    )

    repository.get_by_idempotency_key.assert_not_called()
    repository.create.assert_called_once()
    runtime.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_with_new_idempotency_key_creates_run() -> None:
    repository = _repository()
    repository.get_by_idempotency_key.return_value = None

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            user_id="user-1",
        ),
        idempotency_key="request-123",
    )

    repository.get_by_idempotency_key.assert_called_once_with(
        None,
        "user-1",
        "request-123",
    )

    created = repository.create.call_args.args[0]
    assert created.idempotency_key == "request-123"
    runtime.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_replays_completed_idempotent_run() -> None:
    repository = _repository()

    existing = _idempotent_run(
        output={"answer": "persisted"},
    )
    repository.get_by_idempotency_key.return_value = existing

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            metadata={"request": "same"},
        ),
        idempotency_key="request-123",
    )

    assert result.run_id == "existing-run"
    assert result.response == AgentResponse(
        agent_name="enterprise-analyst",
        output={"answer": "persisted"},
        session_id="session-1",
        metadata={"source": "persisted"},
    )
    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_reuses_running_idempotent_run_reference() -> None:
    repository = _repository()

    existing = _idempotent_run(
        status=AgentRunStatus.RUNNING,
        output=None,
    )
    repository.get_by_idempotency_key.return_value = existing

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            metadata={"request": "same"},
        ),
        idempotency_key="request-123",
    )

    assert result.run_id == "existing-run"
    assert result.response.agent_name == "enterprise-analyst"
    assert result.response.output is None
    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_reuses_pending_idempotent_run_reference() -> None:
    repository = _repository()

    existing = _idempotent_run(
        status=AgentRunStatus.PENDING,
        output=None,
    )
    repository.get_by_idempotency_key.return_value = existing

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            metadata={"request": "same"},
        ),
        idempotency_key="request-123",
    )

    assert result.run_id == "existing-run"
    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_rejects_idempotency_key_for_different_request() -> None:
    repository = _repository()

    repository.get_by_idempotency_key.return_value = _idempotent_run()

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentRunIdempotencyConflictError,
        match="different request",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="A different question",
                session_id="session-1",
                user_id="user-1",
                metadata={"request": "same"},
            ),
            idempotency_key="request-123",
        )

    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_rejects_idempotency_key_for_different_agent() -> None:
    repository = _repository()

    repository.get_by_idempotency_key.return_value = _idempotent_run(
        agent_name="different-agent",
    )

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentRunIdempotencyConflictError,
        match="different agent",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
                metadata={"request": "same"},
            ),
            idempotency_key="request-123",
        )

    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_surfaces_previous_failed_idempotent_run() -> None:
    repository = _repository()

    repository.get_by_idempotency_key.return_value = _idempotent_run(
        status=AgentRunStatus.FAILED,
    )

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="previously failed: persisted failure",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
                metadata={"request": "same"},
            ),
            idempotency_key="request-123",
        )

    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_requires_user_for_idempotency_key() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        ValueError,
        match="user_id is required",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
            ),
            idempotency_key="request-123",
        )

    repository.get_by_idempotency_key.assert_not_called()
    repository.create.assert_not_called()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_rejects_blank_idempotency_key() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        ValueError,
        match="Idempotency key must not be empty",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                user_id="user-1",
            ),
            idempotency_key="   ",
        )

    repository.get_by_idempotency_key.assert_not_called()
    repository.create.assert_not_called()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_race_on_idempotent_create_reloads_existing_run() -> None:
    repository = _repository()

    existing = _idempotent_run()

    repository.get_by_idempotency_key.side_effect = [
        None,
        existing,
    ]
    repository.create.side_effect = DuplicateAgentRunError("agent run already exists")

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            metadata={"request": "same"},
        ),
        idempotency_key="request-123",
    )

    assert result.run_id == "existing-run"
    runtime.run.assert_not_awaited()
    assert repository.get_by_idempotency_key.call_count == 2
    repository.create.assert_called_once()


@pytest.mark.asyncio
async def test_execute_rejects_idempotency_key_for_different_session() -> None:
    repository = _repository()

    repository.get_by_idempotency_key.return_value = _idempotent_run(
        output="persisted",
    )

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentRunIdempotencyConflictError,
        match="different request",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="different-session",
                user_id="user-1",
                metadata={"request": "same"},
            ),
            idempotency_key="request-123",
        )

    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


def _step(
    *,
    run_id: str = "run-123",
    step_id: str = "step-1",
    step_index: int = 0,
    status: AgentRunStepStatus = AgentRunStepStatus.COMPLETED,
) -> AgentRunStep:
    return AgentRunStep(
        run_id=run_id,
        step_id=step_id,
        step_index=step_index,
        step_type="tool_execution",
        status=status,
        attempt=1,
        tool_name="vehicle_query",
        call_id="call-1",
        input={"query": "vehicle events"},
        output={"rows": 3},
        metadata={"source": "test"},
    )


def test_list_steps_delegates_to_step_repository() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    steps_repository = Mock(spec=AgentRunStepsRepository)
    steps = [
        _step(step_id="step-1", step_index=0),
        _step(step_id="step-2", step_index=1),
    ]
    steps_repository.list.return_value = steps

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    result = service.list_steps(
        "run-123",
        tenant_id="tenant-acme",
        principal="api_key:test-owner",
        status=AgentRunStepStatus.COMPLETED,
        limit=25,
    )

    assert result == steps
    repository.get_for_tenant.assert_called_once_with(
        "run-123",
        "tenant-acme",
    )
    steps_repository.list.assert_called_once_with(
        "run-123",
        status=AgentRunStepStatus.COMPLETED,
        limit=25,
    )


def test_list_steps_raises_for_missing_run() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = None

    steps_repository = Mock(spec=AgentRunStepsRepository)

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    with pytest.raises(
        LookupError,
        match="Agent run 'missing-run' was not found.",
    ):
        service.list_steps(
            "missing-run",
            tenant_id="tenant-acme",
            principal="api_key:test-owner",
        )

    repository.get_for_tenant.assert_called_once_with(
        "missing-run",
        "tenant-acme",
    )
    steps_repository.list.assert_not_called()


def test_list_steps_requires_step_repository() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent run steps repository is not configured.",
    ):
        service.list_steps(
            "run-123",
            tenant_id="tenant-acme",
            principal="api_key:test-owner",
        )


def test_get_step_delegates_to_step_repository() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    steps_repository = Mock(spec=AgentRunStepsRepository)
    step = _step()
    steps_repository.get.return_value = step

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    result = service.get_step(
        "run-123",
        "step-1",
        tenant_id="tenant-acme",
        principal="api_key:test-owner",
    )

    assert result is step
    repository.get_for_tenant.assert_called_once_with(
        "run-123",
        "tenant-acme",
    )
    steps_repository.get.assert_called_once_with(
        "run-123",
        "step-1",
    )


def test_get_step_returns_none_for_missing_step() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    steps_repository = Mock(spec=AgentRunStepsRepository)
    steps_repository.get.return_value = None

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    result = service.get_step(
        "run-123",
        "missing-step",
        tenant_id="tenant-acme",
        principal="api_key:test-owner",
    )

    assert result is None
    steps_repository.get.assert_called_once_with(
        "run-123",
        "missing-step",
    )


def test_get_step_raises_for_missing_run() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = None

    steps_repository = Mock(spec=AgentRunStepsRepository)

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    with pytest.raises(
        LookupError,
        match="Agent run 'missing-run' was not found.",
    ):
        service.get_step(
            "missing-run",
            "step-1",
            tenant_id="tenant-acme",
            principal="api_key:test-owner",
        )

    repository.get_for_tenant.assert_called_once_with(
        "missing-run",
        "tenant-acme",
    )
    steps_repository.get.assert_not_called()


def test_get_step_requires_step_repository() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent run steps repository is not configured.",
    ):
        service.get_step(
            "run-123",
            "step-1",
            tenant_id="tenant-acme",
            principal="api_key:test-owner",
        )
