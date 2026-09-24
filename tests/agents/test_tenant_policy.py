from dataclasses import FrozenInstanceError

import pytest

from ai_platform.agents.policy import (
    OutputGovernanceDecision,
    PolicyViolationError,
    TenantPolicy,
    TenantPolicyEngine,
)


@pytest.fixture
def engine() -> TenantPolicyEngine:
    engine = TenantPolicyEngine()
    engine.register_policy(
        TenantPolicy(
            tenant_id="tenant_a",
            allowed_tools=frozenset({"query_db", "read_file"}),
            blocked_tools=frozenset({"delete_db"}),
            allowed_mcp_servers=frozenset({"server_1"}),
            max_tokens_per_run=100_000,
        )
    )
    return engine


def test_allowed_tool_and_mcp_server(engine: TenantPolicyEngine) -> None:
    engine.validate_tool_execution(
        "tenant_a",
        "query_db",
        "server_1",
    )


def test_blocked_tool_is_rejected(engine: TenantPolicyEngine) -> None:
    with pytest.raises(PolicyViolationError, match="explicitly blocked"):
        engine.validate_tool_execution(
            "tenant_a",
            "delete_db",
            "server_1",
        )


def test_unauthorized_tool_is_rejected(engine: TenantPolicyEngine) -> None:
    with pytest.raises(PolicyViolationError, match="not allowed"):
        engine.validate_tool_execution(
            "tenant_a",
            "admin_tool",
            "server_1",
        )


def test_unauthorized_mcp_server_is_rejected(
    engine: TenantPolicyEngine,
) -> None:
    with pytest.raises(PolicyViolationError, match="not authorized"):
        engine.validate_tool_execution(
            "tenant_a",
            "query_db",
            "untrusted_server",
        )


def test_unregistered_tenant_is_rejected(
    engine: TenantPolicyEngine,
) -> None:
    with pytest.raises(PolicyViolationError, match="No policy registered"):
        engine.validate_tool_execution(
            "tenant_unknown",
            "query_db",
        )


def test_policy_is_immutable() -> None:
    policy = TenantPolicy(
        tenant_id="tenant_a",
        allowed_tools=frozenset({"query_db"}),
    )

    with pytest.raises(AttributeError):
        policy.tenant_id = "tenant_b"

    with pytest.raises(AttributeError):
        policy.allowed_tools.add("delete_db")


def test_policy_rejects_negative_token_limit() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TenantPolicy(
            tenant_id="tenant_a",
            max_tokens_per_run=-1,
        )


def test_model_governance_decision_requires_model() -> None:
    from ai_platform.agents.policy import ModelGovernanceDecision

    with pytest.raises(ValueError, match="effective_model"):
        ModelGovernanceDecision(effective_model="")


def test_model_governance_decision_accepts_provider_and_policy_metadata() -> None:
    from ai_platform.agents.policy import ModelGovernanceDecision

    decision = ModelGovernanceDecision(
        effective_model="gpt-5",
        effective_provider="openai",
        policy_id="enterprise-model-policy",
        policy_version="v7",
    )

    assert decision.effective_model == "gpt-5"
    assert decision.effective_provider == "openai"
    assert decision.policy_id == "enterprise-model-policy"
    assert decision.policy_version == "v7"


def test_model_governance_decision_is_immutable() -> None:
    from ai_platform.agents.policy import ModelGovernanceDecision

    decision = ModelGovernanceDecision(effective_model="gpt-5")

    with pytest.raises(AttributeError):
        decision.effective_model = "gpt-4o"


def test_tenant_policy_model_governance_defaults_to_unrestricted() -> None:
    policy = TenantPolicy(tenant_id="tenant-a")

    assert policy.allowed_models is None
    assert policy.allowed_providers is None
    assert policy.policy_id is None
    assert policy.policy_version is None


def test_tenant_policy_accepts_model_governance_configuration() -> None:
    policy = TenantPolicy(
        tenant_id="tenant-a",
        allowed_models=frozenset({"gpt-5", "claude-sonnet-4"}),
        allowed_providers=frozenset({"openai", "anthropic"}),
        policy_id="enterprise-model-policy",
        policy_version="v7",
    )

    assert policy.allowed_models == frozenset({"gpt-5", "claude-sonnet-4"})
    assert policy.allowed_providers == frozenset({"openai", "anthropic"})
    assert policy.policy_id == "enterprise-model-policy"
    assert policy.policy_version == "v7"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("allowed_models", frozenset({""})),
        ("allowed_providers", frozenset({""})),
    ],
)
def test_tenant_policy_rejects_empty_model_governance_values(
    field_name: str,
    value: frozenset[str],
) -> None:
    with pytest.raises(ValueError, match=field_name):
        TenantPolicy(
            tenant_id="tenant-a",
            **{field_name: value},
        )


@pytest.mark.parametrize(
    "field_name",
    ["policy_id", "policy_version"],
)
def test_tenant_policy_rejects_empty_policy_metadata(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        TenantPolicy(
            tenant_id="tenant-a",
            **{field_name: ""},
        )


def test_agent_request_carries_model_governance_decision() -> None:
    from ai_platform.agents.models import AgentRequest
    from ai_platform.agents.policy import ModelGovernanceDecision

    decision = ModelGovernanceDecision(
        effective_model="gpt-5",
        effective_provider="openai",
        policy_id="enterprise-model-policy",
        policy_version="v7",
    )

    request = AgentRequest(
        input="Explain the platform",
        tenant_id="tenant-a",
        model_governance=decision,
    )

    assert request.model_governance == decision
    assert request.model_governance.effective_model == "gpt-5"


def test_output_governance_decision_is_immutable() -> None:
    decision = OutputGovernanceDecision(
        allowed=True,
        redacted_output="safe output",
        policy_id="policy-output",
        policy_version="v1",
    )

    assert decision.redacted_output == "safe output"

    with pytest.raises(FrozenInstanceError):
        decision.redacted_output = "changed"


def test_tenant_policy_output_governance_defaults_disabled() -> None:
    policy = TenantPolicy(tenant_id="tenant-a")

    assert policy.output_governance_enabled is False
    assert policy.blocked_output_patterns == frozenset()
    assert policy.redact_output_patterns == frozenset()


def test_tenant_policy_engine_blocks_output_before_redaction() -> None:
    engine = TenantPolicyEngine()

    engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            policy_id="policy-output",
            policy_version="v3",
            output_governance_enabled=True,
            blocked_output_patterns=frozenset({r"FORBIDDEN"}),
            redact_output_patterns=frozenset({r"secret=[A-Za-z0-9_]+"}),
        )
    )

    decision = engine.evaluate_output(
        "tenant-a",
        "FORBIDDEN secret=abc123",
    )

    assert decision.allowed is False
    assert decision.redacted_output == ""
    assert decision.policy_id == "policy-output"
    assert decision.policy_version == "v3"


def test_tenant_policy_engine_redacts_allowed_output() -> None:
    engine = TenantPolicyEngine()

    engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            policy_id="policy-output",
            policy_version="v3",
            output_governance_enabled=True,
            redact_output_patterns=frozenset({r"secret=[A-Za-z0-9_]+"}),
        )
    )

    raw_output = "The result is secret=abc123."

    decision = engine.evaluate_output("tenant-a", raw_output)

    assert decision.allowed is True
    assert decision.redacted_output == "The result is ********."
    assert decision.governance_metadata["redacted"] is True
    assert "abc123" not in decision.redacted_output


def test_tenant_policy_engine_preserves_output_when_governance_disabled() -> None:
    engine = TenantPolicyEngine()

    engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            output_governance_enabled=False,
            redact_output_patterns=frozenset({r"secret=\S+"}),
        )
    )

    raw_output = "secret=abc123"

    decision = engine.evaluate_output("tenant-a", raw_output)

    assert decision.allowed is True
    assert decision.redacted_output == raw_output


def test_tenant_policy_engine_allows_unmapped_tenant() -> None:
    engine = TenantPolicyEngine()

    raw_output = "secret=abc123"

    decision = engine.evaluate_output("unknown-tenant", raw_output)

    assert decision.allowed is True
    assert decision.redacted_output == raw_output
