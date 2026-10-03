from ai_platform.agents.models import AgentRequest
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot


def test_snapshot_round_trip_preserves_tenant_identity() -> None:
    request = AgentRequest(
        input="retrieve customer policy",
        session_id="session-123",
        user_id="user-123",
        principal="principal-123",
        tenant_id="tenant-acme",
        memory_namespace="tenant-acme-memory",
        metadata={"tenant_id": "attacker-controlled-metadata"},
    )

    snapshot = AgentRunRequestSnapshot.from_request(request)

    assert snapshot.tenant_id == "tenant-acme"
    assert snapshot.metadata["tenant_id"] == "attacker-controlled-metadata"

    recovered = snapshot.to_request(
        session_id=request.session_id,
        user_id=request.user_id,
        principal=request.principal,
    )

    assert recovered.tenant_id == "tenant-acme"
    assert recovered.metadata["tenant_id"] == "attacker-controlled-metadata"


def test_snapshot_with_null_tenant_remains_backward_compatible() -> None:
    snapshot = AgentRunRequestSnapshot(
        input="historical request",
        principal="historical-principal",
        tenant_id=None,
    )

    recovered = snapshot.to_request(
        session_id="historical-session",
        user_id="historical-user",
        principal="historical-principal",
    )

    assert recovered.tenant_id is None
    assert recovered.input == "historical request"
    assert recovered.principal == "historical-principal"


def test_snapshot_round_trip_preserves_max_tokens_per_run() -> None:
    from ai_platform.agents.budget import ExecutionBudget

    request = AgentRequest(
        input="retrieve customer policy",
        session_id="session-budget",
        user_id="user-budget",
        principal="principal-budget",
        tenant_id="tenant-acme",
        execution_budget=ExecutionBudget(
            max_llm_calls=5,
            max_tool_calls=10,
            max_tool_rounds=2,
            max_duration_seconds=120.0,
            max_tokens_per_run=20_000,
        ),
    )

    snapshot = AgentRunRequestSnapshot.from_request(request)

    assert snapshot.execution_budget is not None
    assert snapshot.execution_budget["max_tokens_per_run"] == 20_000

    recovered = snapshot.to_request(
        session_id=request.session_id,
        user_id=request.user_id,
        principal=request.principal,
    )

    assert recovered.execution_budget is not None
    assert recovered.execution_budget.max_tokens_per_run == 20_000


def test_snapshot_round_trip_preserves_unlimited_max_tokens_per_run() -> None:
    from ai_platform.agents.budget import ExecutionBudget

    request = AgentRequest(
        input="historical request",
        session_id="session-unlimited",
        user_id="user-unlimited",
        principal="principal-unlimited",
        tenant_id="tenant-acme",
        execution_budget=ExecutionBudget(
            max_tokens_per_run=None,
        ),
    )

    snapshot = AgentRunRequestSnapshot.from_request(request)

    assert snapshot.execution_budget is not None
    assert snapshot.execution_budget["max_tokens_per_run"] is None

    recovered = snapshot.to_request(
        session_id=request.session_id,
        user_id=request.user_id,
        principal=request.principal,
    )

    assert recovered.execution_budget is not None
    assert recovered.execution_budget.max_tokens_per_run is None


def test_snapshot_round_trip_preserves_model_governance_decision() -> None:
    from ai_platform.agents.policy import ModelGovernanceDecision

    decision = ModelGovernanceDecision(
        effective_model="gpt-5",
        effective_provider="openai",
        policy_id="tenant-ai-policy",
        policy_version="v7",
    )

    request = AgentRequest(
        input="Use the governed model.",
        session_id="session-model-governance",
        user_id="user-model-governance",
        principal="principal-model-governance",
        tenant_id="tenant-acme",
        model_governance=decision,
    )

    snapshot = AgentRunRequestSnapshot.from_request(request)

    assert snapshot.model_governance == {
        "effective_model": "gpt-5",
        "effective_provider": "openai",
        "policy_id": "tenant-ai-policy",
        "policy_version": "v7",
    }

    recovered = snapshot.to_request(
        session_id=request.session_id,
        user_id=request.user_id,
        principal=request.principal,
    )

    assert recovered.model_governance == decision


def test_snapshot_without_model_governance_remains_backward_compatible() -> None:
    snapshot = AgentRunRequestSnapshot(
        input="historical request",
        principal="historical-principal",
        tenant_id="tenant-acme",
    )

    recovered = snapshot.to_request(
        session_id="historical-session",
        user_id="historical-user",
        principal="historical-principal",
    )

    assert recovered.model_governance is None


def test_snapshot_round_trip_preserves_effective_agent_governance() -> None:
    from ai_platform.agents.policy import EffectiveAgentGovernance

    governance = EffectiveAgentGovernance(
        tenant_id="tenant-acme",
        policy_id="enterprise-ai-policy",
        policy_version="v3",
        effective_model="gpt-4.1-mini",
        effective_provider="openai",
        max_tokens_per_run=10_000,
    )

    request = AgentRequest(
        input="Run governed analysis.",
        session_id="session-governance",
        user_id="user-governance",
        principal="principal-governance",
        tenant_id="tenant-acme",
        effective_governance=governance,
    )

    snapshot = AgentRunRequestSnapshot.from_request(request)

    assert snapshot.effective_governance == {
        "tenant_id": "tenant-acme",
        "policy_id": "enterprise-ai-policy",
        "policy_version": "v3",
        "effective_model": "gpt-4.1-mini",
        "effective_provider": "openai",
        "max_tokens_per_run": 10_000,
    }

    recovered = snapshot.to_request(
        session_id=request.session_id,
        user_id=request.user_id,
        principal=request.principal,
    )

    assert recovered.effective_governance == governance


def test_snapshot_without_effective_governance_remains_backward_compatible() -> None:
    snapshot = AgentRunRequestSnapshot(
        input="historical request",
        principal="historical-principal",
        tenant_id="tenant-acme",
    )

    recovered = snapshot.to_request(
        session_id="historical-session",
        user_id="historical-user",
        principal="historical-principal",
    )

    assert recovered.effective_governance is None
