from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from ai_platform.agents.evaluation.answer_evaluation import AgentAnswerEvaluation
from ai_platform.agents.evaluation.models import AgentEvaluationMetrics
from ai_platform.agents.evaluation.policy import (
    AgentEvaluationPolicy,
    AgentQualityGateResult,
)
from ai_platform.agents.evaluation.run import (
    AgentEvaluationLineage,
    AgentEvaluationRun,
)
from app.control_plane.dependencies import (
    get_agent_evaluation_application_service,
)
from app.control_plane.routes.agents import router


class IdentityMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        tenant_id: str | None = "tenant-1",
        principal: str | None = "user-1",
    ) -> None:
        super().__init__(app)
        self.tenant_id = tenant_id
        self.principal = principal

    async def dispatch(self, request, call_next):
        request.state.tenant_id = self.tenant_id
        request.state.principal = self.principal
        return await call_next(request)


class FakeAgentEvaluationApplicationService:
    def __init__(
        self,
        *,
        evaluation: AgentEvaluationRun | None = None,
        evaluations: list[AgentEvaluationRun] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.evaluation = evaluation or make_evaluation()
        self.evaluations = evaluations if evaluations is not None else [self.evaluation]
        self.error = error
        self.evaluate_calls: list[tuple] = []
        self.list_calls: list[tuple] = []

    async def evaluate_run(
        self,
        run_id: str,
        *,
        tenant_id: str,
        principal: str,
        policy: AgentEvaluationPolicy,
        expected_answer: str | None = None,
    ) -> AgentEvaluationRun:
        self.evaluate_calls.append(
            (
                run_id,
                tenant_id,
                principal,
                policy,
            )
        )

        if self.error is not None:
            raise self.error

        return self.evaluation

    def list_evaluations(
        self,
        run_id: str,
        *,
        tenant_id: str,
        principal: str,
        limit: int = 100,
    ) -> list[AgentEvaluationRun]:
        self.list_calls.append(
            (
                run_id,
                tenant_id,
                principal,
                limit,
            )
        )

        if self.error is not None:
            raise self.error

        return self.evaluations


def make_evaluation(
    *,
    evaluation_run_id: str = "evaluation-1",
    tenant_id: str = "tenant-1",
    passed: bool = True,
    policy_id: str | None = None,
    policy_version: str | None = None,
) -> AgentEvaluationRun:
    return AgentEvaluationRun(
        evaluation_run_id=evaluation_run_id,
        created_at=datetime(2026, 9, 24, 10, 0, tzinfo=UTC),
        lineage=AgentEvaluationLineage(
            evaluated_run_id="run-1",
            agent_name="vehicle-agent",
            agent_version="1.2.3",
            tenant_id=tenant_id,
        ),
        metrics=AgentEvaluationMetrics(
            execution_time_ms=125.5,
            steps_total=3,
            tool_calls_total=2,
            tool_calls_successful=2,
            tool_calls_failed=0,
            invalid_tool_calls=0,
            governance_denials=0,
            task_completed=True,
            rag_queries_total=2,
            rag_sources_retrieved_total=5,
            grounding_evaluated=True,
            grounding_supported=True,
            grounding_support_ratio=1.0,
            grounding_supported_sources_total=2,
            grounding_source_candidates_total=5,
            grounding_method="lexical_sentence_support_v1",
            semantic_grounding_evaluated=True,
            semantic_grounding_score=0.93,
            semantic_grounding_passed=True,
            semantic_grounding_method="llm_grounding_judge_v1",
            semantic_grounding_evaluator_model="gpt-4.1-mini",
            semantic_grounding_evaluator_provider="openai",
        ),
        policy=AgentEvaluationPolicy(
            policy_id=policy_id,
            policy_version=policy_version,
            max_execution_time_ms=1000,
            max_steps_per_run=10,
            max_invalid_tool_calls=0,
            allow_governance_denials=False,
            require_task_completed=True,
            name="default-agent-quality",
        ),
        answer_evaluation=AgentAnswerEvaluation(
            evaluated=True,
            exact_match=False,
            semantic_evaluated=True,
            semantic_score=0.94,
            semantic_passed=True,
            semantic_method="llm_judge_v1",
            evaluator_model="gpt-4.1-mini",
            evaluator_provider="openai",
        ),
        quality_gate=AgentQualityGateResult(
            passed=passed,
            violations=() if passed else ("quality failure",),
        ),
    )


def build_client(
    service: FakeAgentEvaluationApplicationService,
    *,
    tenant_id: str | None = "tenant-1",
    principal: str | None = "user-1",
) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(
        IdentityMiddleware,
        tenant_id=tenant_id,
        principal=principal,
    )
    app.dependency_overrides[get_agent_evaluation_application_service] = lambda: service

    return TestClient(app)


def test_create_agent_run_evaluation_returns_evaluation_artifact() -> None:
    service = FakeAgentEvaluationApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/run-1/evaluations",
        json={
            "max_execution_time_ms": 5000,
            "max_steps_per_run": 10,
            "max_invalid_tool_calls": 0,
            "allow_governance_denials": False,
            "require_task_completed": True,
            "name": "route-quality-policy",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["evaluation_run_id"] == "evaluation-1"
    assert body["passed"] is True
    assert body["lineage"] == {
        "evaluated_run_id": "run-1",
        "agent_name": "vehicle-agent",
        "agent_version": "1.2.3",
        "tenant_id": "tenant-1",
        "effective_model": None,
        "effective_provider": None,
        "model_policy_id": None,
        "model_policy_version": None,
    }
    assert body["metrics"]["steps_total"] == 3
    assert body["metrics"]["tool_calls_total"] == 2
    assert body["metrics"]["rag_queries_total"] == 2
    assert body["metrics"]["rag_sources_retrieved_total"] == 5
    assert body["metrics"]["grounding_evaluated"] is True
    assert body["metrics"]["grounding_supported"] is True
    assert body["metrics"]["grounding_support_ratio"] == 1.0
    assert body["metrics"]["grounding_supported_sources_total"] == 2
    assert body["metrics"]["grounding_source_candidates_total"] == 5
    assert body["metrics"]["grounding_method"] == "lexical_sentence_support_v1"
    assert body["metrics"]["semantic_grounding_evaluated"] is True
    assert body["metrics"]["semantic_grounding_score"] == 0.93
    assert body["metrics"]["semantic_grounding_passed"] is True
    assert body["metrics"]["semantic_grounding_method"] == "llm_grounding_judge_v1"
    assert body["metrics"]["semantic_grounding_evaluator_model"] == "gpt-4.1-mini"
    assert body["metrics"]["semantic_grounding_evaluator_provider"] == "openai"
    assert body["metrics"]["task_completed"] is True
    assert body["policy"]["name"] == "default-agent-quality"
    assert body["quality_gate"]["passed"] is True
    assert body["answer_evaluation"] == {
        "evaluated": True,
        "exact_match": False,
        "normalization": "whitespace_casefold",
        "semantic_evaluated": True,
        "semantic_score": 0.94,
        "semantic_passed": True,
        "semantic_method": "llm_judge_v1",
        "evaluator_model": "gpt-4.1-mini",
        "evaluator_provider": "openai",
    }

    assert len(service.evaluate_calls) == 1
    run_id, tenant_id, principal, policy = service.evaluate_calls[0]

    assert run_id == "run-1"
    assert tenant_id == "tenant-1"
    assert principal == "user-1"
    assert policy.max_execution_time_ms == 5000
    assert policy.max_steps_per_run == 10
    assert policy.max_invalid_tool_calls == 0
    assert policy.name == "route-quality-policy"


def test_create_agent_run_evaluation_preserves_policy_identity_and_version() -> None:
    service = FakeAgentEvaluationApplicationService(
        evaluation=make_evaluation(
            policy_id="rag-quality",
            policy_version="1.0",
        )
    )
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/run-1/evaluations",
        json={
            "policy_id": "rag-quality",
            "policy_version": "1.0",
            "name": "route-quality-policy",
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert body["policy"]["policy_id"] == "rag-quality"
    assert body["policy"]["policy_version"] == "1.0"

    assert len(service.evaluate_calls) == 1
    _, _, _, policy = service.evaluate_calls[0]

    assert policy.policy_id == "rag-quality"
    assert policy.policy_version == "1.0"


def test_create_agent_run_evaluation_rejects_policy_id_without_version() -> None:
    service = FakeAgentEvaluationApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/run-1/evaluations",
        json={
            "policy_id": "rag-quality",
            "name": "route-quality-policy",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == (
        "Value error, policy_id and policy_version must be provided together."
    )
    assert service.evaluate_calls == []


def test_create_agent_run_evaluation_rejects_policy_version_without_id() -> None:
    service = FakeAgentEvaluationApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/run-1/evaluations",
        json={
            "policy_version": "1.0",
            "name": "route-quality-policy",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == (
        "Value error, policy_id and policy_version must be provided together."
    )
    assert service.evaluate_calls == []


def test_create_agent_run_evaluation_requires_identity_context() -> None:
    service = FakeAgentEvaluationApplicationService()
    client = build_client(
        service,
        tenant_id=None,
        principal=None,
    )

    response = client.post(
        "/api/v1/agents/runs/run-1/evaluations",
        json={},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Both tenant_id and principal are required for evaluation."
    )
    assert service.evaluate_calls == []


def test_create_agent_run_evaluation_maps_permission_error_to_403() -> None:
    service = FakeAgentEvaluationApplicationService(
        error=PermissionError("principal is not authorized")
    )
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/run-1/evaluations",
        json={},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "principal is not authorized"


def test_create_agent_run_evaluation_maps_missing_run_to_404() -> None:
    service = FakeAgentEvaluationApplicationService(
        error=LookupError("agent run not found: run-missing")
    )
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/run-missing/evaluations",
        json={},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "agent run not found: run-missing"


def test_create_agent_run_evaluation_rejects_invalid_policy() -> None:
    service = FakeAgentEvaluationApplicationService()
    client = build_client(service)

    response = client.post(
        "/api/v1/agents/runs/run-1/evaluations",
        json={
            "max_execution_time_ms": -1,
        },
    )

    assert response.status_code == 422
    assert service.evaluate_calls == []


def test_list_agent_run_evaluations_returns_evaluations() -> None:
    service = FakeAgentEvaluationApplicationService(
        evaluations=[
            make_evaluation(evaluation_run_id="evaluation-2"),
            make_evaluation(evaluation_run_id="evaluation-1"),
        ]
    )
    client = build_client(service)

    response = client.get(
        "/api/v1/agents/runs/run-1/evaluations?limit=2",
    )

    assert response.status_code == 200
    body = response.json()

    assert body["limit"] == 2
    assert [evaluation["evaluation_run_id"] for evaluation in body["evaluations"]] == [
        "evaluation-2",
        "evaluation-1",
    ]

    for evaluation in body["evaluations"]:
        assert evaluation["metrics"]["semantic_grounding_evaluated"] is True
        assert evaluation["metrics"]["semantic_grounding_score"] == 0.93
        assert evaluation["metrics"]["semantic_grounding_passed"] is True
        assert evaluation["metrics"]["semantic_grounding_method"] == ("llm_grounding_judge_v1")
        assert evaluation["metrics"]["semantic_grounding_evaluator_model"] == ("gpt-4.1-mini")
        assert evaluation["metrics"]["semantic_grounding_evaluator_provider"] == ("openai")

    assert service.list_calls == [
        ("run-1", "tenant-1", "user-1", 2),
    ]


def test_list_agent_run_evaluations_requires_identity_context() -> None:
    service = FakeAgentEvaluationApplicationService()
    client = build_client(
        service,
        tenant_id=None,
        principal=None,
    )

    response = client.get(
        "/api/v1/agents/runs/run-1/evaluations",
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "Both tenant_id and principal are required for evaluation."
    )
    assert service.list_calls == []


def test_list_agent_run_evaluations_maps_permission_error_to_403() -> None:
    service = FakeAgentEvaluationApplicationService(
        error=PermissionError("principal is not authorized")
    )
    client = build_client(service)

    response = client.get(
        "/api/v1/agents/runs/run-1/evaluations",
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "principal is not authorized"


def test_list_agent_run_evaluations_maps_missing_run_to_404() -> None:
    service = FakeAgentEvaluationApplicationService(
        error=LookupError("agent run not found: run-missing")
    )
    client = build_client(service)

    response = client.get(
        "/api/v1/agents/runs/run-missing/evaluations",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "agent run not found: run-missing"


def test_list_agent_run_evaluations_rejects_invalid_limit() -> None:
    service = FakeAgentEvaluationApplicationService()
    client = build_client(service)

    response = client.get(
        "/api/v1/agents/runs/run-1/evaluations?limit=0",
    )

    assert response.status_code == 422
    assert service.list_calls == []
