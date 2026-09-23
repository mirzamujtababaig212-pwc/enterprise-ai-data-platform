import pytest

from ai_platform.agents.policy import (
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
