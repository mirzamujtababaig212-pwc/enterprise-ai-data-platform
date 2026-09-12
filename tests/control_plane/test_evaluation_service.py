from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from app.control_plane.evaluation_service import EvaluationExecutionService
from rag.evaluation.composite_release import (
    CompositeEvaluationReleaseDecision,
)
from rag.evaluation.external.release import ExternalEvaluationReleasePolicy
from rag.evaluation.run import RetrievalEvaluationRun


def _workflow_result():
    from rag.evaluation.workflow import RetrievalEvaluationWorkflowResult

    return Mock(spec=RetrievalEvaluationWorkflowResult)


def _run() -> RetrievalEvaluationRun:
    return Mock(spec=RetrievalEvaluationRun, run_id="run-1")


def _decision() -> CompositeEvaluationReleaseDecision:
    return Mock(
        spec=CompositeEvaluationReleaseDecision,
        run_id="run-1",
    )


@pytest.mark.asyncio
async def test_execute_persists_run_and_release_decision_in_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()
    run_store = Mock()
    run_store.save = AsyncMock()
    release_store = Mock()
    release_store.save = AsyncMock()

    workflow_result = Mock()
    workflow_result.run = _run()
    workflow_result.release_decision = _decision()

    workflow_run = Mock(return_value=workflow_result)
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.CompositeEvaluationWorkflow.run",
        workflow_run,
    )

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    result = await service.execute(
        result=_workflow_result(),
        run_id="run-1",
        created_at=datetime.now(UTC),
        external_release_policy=ExternalEvaluationReleasePolicy(),
    )

    assert result.run is workflow_result.run
    assert result.release_decision is workflow_result.release_decision

    run_store.save.assert_awaited_once_with(
        workflow_result.run,
        commit=False,
    )
    release_store.save.assert_awaited_once_with(
        workflow_result.release_decision,
        commit=False,
    )

    session.commit.assert_called_once()
    session.rollback.assert_not_called()

    workflow_run.assert_called_once()


@pytest.mark.asyncio
async def test_execute_rolls_back_when_release_decision_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()

    run_store = Mock()
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock(
        side_effect=RuntimeError("release persistence failed"),
    )

    workflow_result = Mock()
    workflow_result.run = _run()
    workflow_result.release_decision = _decision()

    monkeypatch.setattr(
        "app.control_plane.evaluation_service.CompositeEvaluationWorkflow.run",
        Mock(return_value=workflow_result),
    )

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    with pytest.raises(RuntimeError, match="release persistence failed"):
        await service.execute(
            result=_workflow_result(),
            run_id="run-1",
            created_at=datetime.now(UTC),
            external_release_policy=ExternalEvaluationReleasePolicy(),
        )

    run_store.save.assert_awaited_once_with(
        workflow_result.run,
        commit=False,
    )
    release_store.save.assert_awaited_once_with(
        workflow_result.release_decision,
        commit=False,
    )

    session.commit.assert_not_called()
    session.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_execute_rolls_back_when_commit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()
    session.commit.side_effect = RuntimeError("commit failed")

    run_store = Mock()
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock()

    workflow_result = Mock()
    workflow_result.run = _run()
    workflow_result.release_decision = _decision()

    monkeypatch.setattr(
        "app.control_plane.evaluation_service.CompositeEvaluationWorkflow.run",
        Mock(return_value=workflow_result),
    )

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    with pytest.raises(RuntimeError, match="commit failed"):
        await service.execute(
            result=_workflow_result(),
            run_id="run-1",
            created_at=datetime.now(UTC),
            external_release_policy=ExternalEvaluationReleasePolicy(),
        )

    run_store.save.assert_awaited_once_with(
        workflow_result.run,
        commit=False,
    )
    release_store.save.assert_awaited_once_with(
        workflow_result.release_decision,
        commit=False,
    )

    session.commit.assert_called_once()
    session.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_execute_does_not_persist_when_domain_workflow_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()

    run_store = Mock()
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock()

    monkeypatch.setattr(
        "app.control_plane.evaluation_service.CompositeEvaluationWorkflow.run",
        Mock(side_effect=ValueError("invalid evaluation configuration")),
    )

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    with pytest.raises(ValueError, match="invalid evaluation configuration"):
        await service.execute(
            result=_workflow_result(),
            run_id="run-1",
            created_at=datetime.now(UTC),
            external_release_policy=ExternalEvaluationReleasePolicy(),
        )

    run_store.save.assert_not_awaited()
    release_store.save.assert_not_awaited()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()
