from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from ai_platform.agents.models import AgentRequest, AgentResponse

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
    assert repository.update.call_count == 2

    pending = repository.create.call_args.args[0]
    running = repository.update.call_args_list[0].args[0]
    completed = repository.update.call_args_list[1].args[0]

    assert pending.status == AgentRunStatus.PENDING
    assert pending.agent_name == "enterprise-analyst"
    assert pending.session_id == "session-1"
    assert pending.user_id == "user-1"
    assert pending.started_at is None
    assert pending.completed_at is None

    assert running.run_id == pending.run_id
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

    runtime.run.assert_awaited_once_with(
        "enterprise-analyst",
        AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
        ),
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

    runs = [
        repository.create.call_args.args[0],
        repository.update.call_args_list[0].args[0],
        repository.update.call_args_list[1].args[0],
    ]

    assert len({run.run_id for run in runs}) == 1


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
    assert repository.update.call_count == 2

    pending = repository.create.call_args.args[0]
    running = repository.update.call_args_list[0].args[0]
    failed = repository.update.call_args_list[1].args[0]

    assert pending.status == AgentRunStatus.PENDING
    assert running.status == AgentRunStatus.RUNNING

    assert failed.run_id == pending.run_id
    assert failed.status == AgentRunStatus.FAILED
    assert failed.started_at == running.started_at
    assert failed.completed_at is not None
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "agent execution failed"
    assert failed.output is None


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

    failed = repository.update.call_args_list[-1].args[0]

    assert failed.status == AgentRunStatus.FAILED
    assert failed.error_type == "LookupError"
    assert failed.error_message == "agent not found"


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
