from datetime import UTC, datetime

import pytest

from ai_platform.agents.evaluation.models import AgentEvaluationMetrics
from ai_platform.agents.evaluation.policy import (
    AgentEvaluationPolicy,
    AgentQualityGateResult,
)
from ai_platform.agents.evaluation.run import (
    AgentEvaluationLineage,
    AgentEvaluationRun,
)
from app.control_plane.agent_evaluations.in_memory import (
    InMemoryAgentEvaluationRunsRepository,
)
from app.control_plane.agent_evaluations.repository import (
    DuplicateAgentEvaluationRunError,
)


def make_evaluation_run(
    evaluation_run_id: str = "evaluation-1",
    *,
    evaluated_run_id: str = "run-1",
    tenant_id: str | None = "tenant-1",
    created_at: datetime | None = None,
    passed: bool = True,
) -> AgentEvaluationRun:
    metrics = AgentEvaluationMetrics(
        execution_time_ms=125.5,
        steps_total=3,
        tool_calls_total=2,
        tool_calls_successful=2,
        tool_calls_failed=0,
        invalid_tool_calls=0,
        governance_denials=0,
        task_completed=True,
    )

    policy = AgentEvaluationPolicy(
        max_execution_time_ms=1000,
        max_steps_per_run=10,
        max_invalid_tool_calls=0,
        allow_governance_denials=False,
        require_task_completed=True,
        name="default-agent-quality",
    )

    quality_gate = AgentQualityGateResult(
        passed=passed,
        violations=() if passed else ("quality failure",),
    )

    return AgentEvaluationRun(
        evaluation_run_id=evaluation_run_id,
        created_at=created_at or datetime.now(UTC),
        lineage=AgentEvaluationLineage(
            evaluated_run_id=evaluated_run_id,
            agent_name="vehicle-agent",
            agent_version=None,
            tenant_id=tenant_id,
        ),
        metrics=metrics,
        policy=policy,
        quality_gate=quality_gate,
    )


class TestInMemoryAgentEvaluationRunsRepository:
    def test_save_and_get_round_trip(self):
        repository = InMemoryAgentEvaluationRunsRepository()
        run = make_evaluation_run()

        saved = repository.save(run)

        assert saved == run
        assert repository.get("evaluation-1") == run

    def test_duplicate_save_is_rejected(self):
        repository = InMemoryAgentEvaluationRunsRepository()
        run = make_evaluation_run()

        repository.save(run)

        with pytest.raises(
            DuplicateAgentEvaluationRunError,
            match="agent evaluation run already exists: evaluation-1",
        ):
            repository.save(run)

    def test_get_missing_returns_none(self):
        repository = InMemoryAgentEvaluationRunsRepository()

        assert repository.get("missing") is None

    def test_list_orders_newest_first(self):
        repository = InMemoryAgentEvaluationRunsRepository()

        older = make_evaluation_run(
            "evaluation-old",
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        newer = make_evaluation_run(
            "evaluation-new",
            created_at=datetime(2026, 1, 2, tzinfo=UTC),
        )

        repository.save(older)
        repository.save(newer)

        result = repository.list()

        assert [run.evaluation_run_id for run in result] == [
            "evaluation-new",
            "evaluation-old",
        ]

    def test_list_filters_by_evaluated_run(self):
        repository = InMemoryAgentEvaluationRunsRepository()

        repository.save(
            make_evaluation_run(
                "evaluation-1",
                evaluated_run_id="run-1",
            )
        )
        repository.save(
            make_evaluation_run(
                "evaluation-2",
                evaluated_run_id="run-2",
            )
        )

        result = repository.list(evaluated_run_id="run-1")

        assert [run.evaluation_run_id for run in result] == ["evaluation-1"]

    def test_list_filters_by_tenant(self):
        repository = InMemoryAgentEvaluationRunsRepository()

        repository.save(
            make_evaluation_run(
                "evaluation-1",
                tenant_id="tenant-1",
            )
        )
        repository.save(
            make_evaluation_run(
                "evaluation-2",
                tenant_id="tenant-2",
            )
        )

        result = repository.list(tenant_id="tenant-2")

        assert [run.evaluation_run_id for run in result] == ["evaluation-2"]

    def test_list_rejects_non_positive_limit(self):
        repository = InMemoryAgentEvaluationRunsRepository()

        with pytest.raises(
            ValueError,
            match="limit must be greater than zero",
        ):
            repository.list(limit=0)

    def test_count_without_filters(self):
        repository = InMemoryAgentEvaluationRunsRepository()

        repository.save(make_evaluation_run("evaluation-1"))
        repository.save(make_evaluation_run("evaluation-2"))

        assert repository.count() == 2

    def test_count_supports_filters(self):
        repository = InMemoryAgentEvaluationRunsRepository()

        repository.save(
            make_evaluation_run(
                "evaluation-1",
                evaluated_run_id="run-1",
                tenant_id="tenant-1",
            )
        )
        repository.save(
            make_evaluation_run(
                "evaluation-2",
                evaluated_run_id="run-1",
                tenant_id="tenant-2",
            )
        )
        repository.save(
            make_evaluation_run(
                "evaluation-3",
                evaluated_run_id="run-2",
                tenant_id="tenant-1",
            )
        )

        assert repository.count(evaluated_run_id="run-1") == 2
        assert repository.count(tenant_id="tenant-1") == 2
        assert (
            repository.count(
                evaluated_run_id="run-1",
                tenant_id="tenant-1",
            )
            == 1
        )

    def test_clear_removes_all_runs(self):
        repository = InMemoryAgentEvaluationRunsRepository()
        repository.save(make_evaluation_run())

        repository.clear()

        assert repository.count() == 0
        assert repository.get("evaluation-1") is None
