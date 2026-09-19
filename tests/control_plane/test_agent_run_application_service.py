from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

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
    AgentRunAdmissionRejectedError,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
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

    repository.complete_if_owner.side_effect = complete_if_owner
    repository.fail_if_owner.side_effect = fail_if_owner

    return repository


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
        run_id=pending.run_id,
    )


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

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        lease_seconds=60,
    )

    sleep_calls = 0

    async def sleep_once(seconds: float) -> None:
        nonlocal sleep_calls

        assert seconds == 20
        sleep_calls += 1

        if sleep_calls == 2:
            raise asyncio.CancelledError

    with patch(
        "app.control_plane.agent_runs.application_service.asyncio.sleep",
        side_effect=sleep_once,
    ):
        with pytest.raises(asyncio.CancelledError):
            await service._heartbeat_loop(
                run_id="run-heartbeat",
                lease_id="lease-123",
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

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        lease_seconds=60,
    )

    async def sleep_once(seconds: float) -> None:
        assert seconds == 20

    with patch(
        "app.control_plane.agent_runs.application_service.asyncio.sleep",
        side_effect=sleep_once,
    ):
        await service._heartbeat_loop(
            run_id="run-heartbeat",
            lease_id="lease-123",
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

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        lease_seconds=60,
    )

    sleep_calls = 0

    async def sleep_once(seconds: float) -> None:
        nonlocal sleep_calls

        assert seconds == 20
        sleep_calls += 1

        if sleep_calls == 3:
            raise asyncio.CancelledError

    with patch(
        "app.control_plane.agent_runs.application_service.asyncio.sleep",
        side_effect=sleep_once,
    ):
        with pytest.raises(asyncio.CancelledError):
            await service._heartbeat_loop(
                run_id="run-heartbeat",
                lease_id="lease-123",
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
