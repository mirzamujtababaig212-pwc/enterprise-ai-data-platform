from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest

from app.control_plane.evaluation_application_service import (
    EvaluationApplicationService,
)
from app.control_plane.evaluation_service import EvaluationExecutionResult
from rag.evaluation.composite_release import CompositeEvaluationReleaseDecision
from rag.evaluation.external.release import ExternalEvaluationReleasePolicy
from rag.evaluation.policy import RetrievalEvaluationPolicy
from rag.evaluation.run import RetrievalEvaluationRun


def _run() -> RetrievalEvaluationRun:
    return Mock(spec=RetrievalEvaluationRun, run_id="run-1")


def _decision() -> CompositeEvaluationReleaseDecision:
    return Mock(
        spec=CompositeEvaluationReleaseDecision,
        run_id="run-1",
    )


@pytest.mark.asyncio
async def test_execute_builds_registered_dataset_and_delegates_to_execution_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_service = Mock()
    execution_service.execute = AsyncMock()

    run = _run()
    decision = _decision()

    execution_service.execute.return_value = EvaluationExecutionResult(
        run=run,
        release_decision=decision,
    )

    service = EvaluationApplicationService(
        execution_service=execution_service,
    )

    result = await service.execute(
        dataset_name="vehicle-retrieval",
        dataset_version="v2",
        run_id="run-1",
        created_at=datetime.now(UTC),
        evaluation_policy=RetrievalEvaluationPolicy(
            name="vehicle-retrieval-quality-test",
            min_recall_at_k=1.0,
            min_precision_at_k=0.8,
            min_mrr=1.0,
            min_ndcg_at_k=0.95,
        ),
        external_release_policy=ExternalEvaluationReleasePolicy(),
    )

    assert result.run is run
    assert result.release_decision is decision

    execution_service.execute.assert_awaited_once()

    call = execution_service.execute.await_args
    assert call.kwargs["run_id"] == "run-1"
    assert call.kwargs["external_release_policy"] == ExternalEvaluationReleasePolicy()

    delegated_result = call.kwargs["result"]

    assert delegated_result.dataset_name == "vehicle-retrieval"
    assert delegated_result.evaluation is not None
    assert delegated_result.quality_gate is not None
    assert delegated_result.lineage.dataset_name == "vehicle-retrieval"
    assert delegated_result.lineage.dataset_version == "v2"


@pytest.mark.asyncio
async def test_execute_rejects_unknown_dataset_before_persistence() -> None:
    execution_service = Mock()
    execution_service.execute = AsyncMock()

    service = EvaluationApplicationService(
        execution_service=execution_service,
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
            evaluation_policy=RetrievalEvaluationPolicy(
                name="vehicle-retrieval-quality-test",
                min_recall_at_k=1.0,
                min_precision_at_k=0.8,
                min_mrr=1.0,
                min_ndcg_at_k=0.95,
            ),
            external_release_policy=ExternalEvaluationReleasePolicy(),
        )

    execution_service.execute.assert_not_awaited()
