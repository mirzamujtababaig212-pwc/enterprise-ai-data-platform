from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from app.control_plane.evaluation_service import EvaluationExecutionService
from rag.evaluation.composite_release import CompositeEvaluationReleaseDecision
from rag.evaluation.external.release import ExternalEvaluationReleasePolicy
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.run import RetrievalEvaluationRun
from rag.evaluation.run_store import DuplicateEvaluationRunError


def _evaluation_policy() -> RetrievalEvaluationPolicy:
    return RetrievalEvaluationPolicy(
        name="vehicle-retrieval-quality-test",
        min_recall_at_k=1.0,
        min_precision_at_k=0.8,
        min_mrr=1.0,
        min_ndcg_at_k=0.95,
    )


def _dataset_definition():
    definition = Mock()
    definition.name = "vehicle-retrieval"
    definition.version = "v2"
    dataset = Mock()
    dataset.name = "vehicle-retrieval"
    dataset.version = "v2"
    definition.build_dataset.return_value = dataset
    definition.build_embedding_identity.return_value = Mock()
    definition.build_retriever = AsyncMock(return_value=Mock())
    return definition


def _run() -> RetrievalEvaluationRun:
    return Mock(spec=RetrievalEvaluationRun, run_id="run-1")


def _decision() -> CompositeEvaluationReleaseDecision:
    return Mock(
        spec=CompositeEvaluationReleaseDecision,
        run_id="run-1",
    )


def _workflow_result():
    workflow_result = Mock()
    workflow_result.run = _run()
    workflow_result.release_decision = _decision()
    return workflow_result


@pytest.mark.asyncio
async def test_execute_persists_run_and_release_decision_in_one_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()

    run_store = Mock()
    run_store.get = AsyncMock(return_value=None)
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock()

    definition = _dataset_definition()
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.EvaluationDatasetRegistry.get",
        Mock(return_value=definition),
    )

    evaluation_workflow_result = Mock()

    monkeypatch.setattr(
        "app.control_plane.evaluation_service.RetrievalEvaluationWorkflow.run",
        AsyncMock(return_value=evaluation_workflow_result),
    )

    composite_result = _workflow_result()

    composite_workflow_run = Mock(return_value=composite_result)
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.CompositeEvaluationWorkflow.run",
        composite_workflow_run,
    )

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    result = await service.execute(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        run_id="run-1",
        created_at=datetime.now(UTC),
        evaluation_policy=_evaluation_policy(),
        external_release_policy=ExternalEvaluationReleasePolicy(),
    )

    assert result.run is composite_result.run
    assert result.release_decision is composite_result.release_decision

    run_store.save.assert_awaited_once_with(
        composite_result.run,
        commit=False,
    )
    release_store.save.assert_awaited_once_with(
        composite_result.release_decision,
        commit=False,
    )

    session.commit.assert_called_once()
    session.rollback.assert_not_called()

    composite_workflow_run.assert_called_once()


@pytest.mark.asyncio
async def test_execute_rolls_back_when_release_decision_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()

    run_store = Mock()
    run_store.get = AsyncMock(return_value=None)
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock(
        side_effect=RuntimeError("release persistence failed"),
    )

    definition = _dataset_definition()
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.EvaluationDatasetRegistry.get",
        Mock(return_value=definition),
    )

    monkeypatch.setattr(
        "app.control_plane.evaluation_service.RetrievalEvaluationWorkflow.run",
        AsyncMock(return_value=Mock()),
    )

    workflow_result = _workflow_result()

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
            dataset_name="vehicle-retrieval",
            dataset_version="v2",
            run_id="run-1",
            created_at=datetime.now(UTC),
            evaluation_policy=_evaluation_policy(),
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
    run_store.get = AsyncMock(return_value=None)
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock()

    definition = _dataset_definition()
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.EvaluationDatasetRegistry.get",
        Mock(return_value=definition),
    )

    monkeypatch.setattr(
        "app.control_plane.evaluation_service.RetrievalEvaluationWorkflow.run",
        AsyncMock(return_value=Mock()),
    )

    workflow_result = _workflow_result()

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
            dataset_name="vehicle-retrieval",
            dataset_version="v2",
            run_id="run-1",
            created_at=datetime.now(UTC),
            evaluation_policy=_evaluation_policy(),
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
    run_store.get = AsyncMock(return_value=None)
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock()

    definition = _dataset_definition()
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.EvaluationDatasetRegistry.get",
        Mock(return_value=definition),
    )

    monkeypatch.setattr(
        "app.control_plane.evaluation_service.RetrievalEvaluationWorkflow.run",
        AsyncMock(side_effect=ValueError("invalid evaluation configuration")),
    )

    composite_workflow_run = Mock()
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.CompositeEvaluationWorkflow.run",
        composite_workflow_run,
    )

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    with pytest.raises(ValueError, match="invalid evaluation configuration"):
        await service.execute(
            dataset_name="vehicle-retrieval",
            dataset_version="v2",
            run_id="run-1",
            created_at=datetime.now(UTC),
            evaluation_policy=_evaluation_policy(),
            external_release_policy=ExternalEvaluationReleasePolicy(),
        )

    run_store.save.assert_not_awaited()
    release_store.save.assert_not_awaited()
    composite_workflow_run.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_execute_rejects_duplicate_run_id_before_domain_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()

    existing_run = _run()

    run_store = Mock()
    run_store.get = AsyncMock(return_value=existing_run)
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock()

    workflow_run = Mock()
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.RetrievalEvaluationWorkflow.run",
        workflow_run,
    )

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    with pytest.raises(
        DuplicateEvaluationRunError,
        match="evaluation run already exists: run-1",
    ):
        await service.execute(
            dataset_name="vehicle-retrieval",
            dataset_version="v2",
            run_id="run-1",
            created_at=datetime.now(UTC),
            evaluation_policy=_evaluation_policy(),
            external_release_policy=ExternalEvaluationReleasePolicy(),
        )

    run_store.get.assert_awaited_once_with("run-1")
    run_store.save.assert_not_awaited()
    release_store.save.assert_not_awaited()
    workflow_run.assert_not_called()
    session.commit.assert_not_called()
    session.rollback.assert_not_called()


@pytest.mark.asyncio
async def test_execute_resolves_registered_dataset_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Mock()

    run_store = Mock()
    run_store.get = AsyncMock(return_value=None)
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock()

    definition = _dataset_definition()

    registry_get = Mock(return_value=definition)
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.EvaluationDatasetRegistry.get",
        registry_get,
    )

    evaluation_workflow_result = Mock()
    workflow_run = AsyncMock(return_value=evaluation_workflow_result)
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.RetrievalEvaluationWorkflow.run",
        workflow_run,
    )

    composite_result = _workflow_result()
    monkeypatch.setattr(
        "app.control_plane.evaluation_service.CompositeEvaluationWorkflow.run",
        Mock(return_value=composite_result),
    )

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    await service.execute(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        run_id="run-1",
        created_at=datetime.now(UTC),
        evaluation_policy=_evaluation_policy(),
        external_release_policy=ExternalEvaluationReleasePolicy(),
    )

    registry_get.assert_called_once_with(
        name="vehicle-retrieval",
        version="v2",
    )
    definition.build_dataset.assert_called_once()
    definition.build_retriever.assert_awaited_once()
    definition.build_embedding_identity.assert_called_once()
    workflow_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_rejects_unknown_dataset_before_persistence() -> None:
    session = Mock()

    run_store = Mock()
    run_store.get = AsyncMock(return_value=None)
    run_store.save = AsyncMock()

    release_store = Mock()
    release_store.save = AsyncMock()

    service = EvaluationExecutionService(
        session=session,
        run_store=run_store,
        release_decision_store=release_store,
    )

    with pytest.raises(
        ValueError,
        match="evaluation dataset not found",
    ):
        await service.execute(
            dataset_name="does-not-exist",
            dataset_version="v1",
            run_id="run-1",
            created_at=datetime.now(UTC),
            evaluation_policy=_evaluation_policy(),
            external_release_policy=ExternalEvaluationReleasePolicy(),
        )

    run_store.get.assert_awaited_once_with("run-1")
    run_store.save.assert_not_awaited()
    release_store.save.assert_not_awaited()
    session.commit.assert_not_called()
