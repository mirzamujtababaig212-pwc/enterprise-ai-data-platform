from __future__ import annotations

import json

import pytest

from app.config.settings import Settings


def test_external_evaluation_release_is_optional_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "EXTERNAL_EVALUATION_RELEASE_REQUIRED",
        raising=False,
    )

    settings = Settings.from_environment()

    assert settings.external_evaluation_release_required is False


@pytest.mark.parametrize(
    "value",
    [
        "1",
        "true",
        "TRUE",
        "yes",
        "on",
    ],
)
def test_external_evaluation_release_accepts_truthy_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "EXTERNAL_EVALUATION_RELEASE_REQUIRED",
        value,
    )

    settings = Settings.from_environment()

    assert settings.external_evaluation_release_required is True


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "false",
        "FALSE",
        "no",
        "off",
        "",
    ],
)
def test_external_evaluation_release_accepts_falsy_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv(
        "EXTERNAL_EVALUATION_RELEASE_REQUIRED",
        value,
    )

    settings = Settings.from_environment()

    assert settings.external_evaluation_release_required is False


def test_agent_run_lease_duration_defaults_to_60_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "AGENT_RUN_LEASE_DURATION_SECONDS",
        raising=False,
    )

    settings = Settings.from_environment()

    assert settings.agent_run_lease_duration_seconds == 60


def test_agent_run_lease_duration_can_be_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AGENT_RUN_LEASE_DURATION_SECONDS",
        "120",
    )

    settings = Settings.from_environment()

    assert settings.agent_run_lease_duration_seconds == 120


def test_mcp_servers_rejects_non_array_args(monkeypatch) -> None:
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps(
            [
                {
                    "name": "server-a",
                    "transport": "stdio",
                    "command": "python",
                    "args": "server.py",
                }
            ]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="MCP_SERVERS contains an invalid server configuration",
    ):
        Settings.from_environment()


def test_mcp_servers_rejects_non_object_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps(
            [
                {
                    "name": "server-a",
                    "transport": "stdio",
                    "command": "python",
                    "env": [],
                }
            ]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="MCP_SERVERS contains an invalid server configuration",
    ):
        Settings.from_environment()


def test_mcp_servers_rejects_non_object_headers(monkeypatch) -> None:
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps(
            [
                {
                    "name": "server-a",
                    "transport": "streamable-http",
                    "url": "http://127.0.0.1:9000/mcp",
                    "headers": [],
                }
            ]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="MCP_SERVERS contains an invalid server configuration",
    ):
        Settings.from_environment()


def test_mcp_servers_rejects_non_boolean_verify_ssl(monkeypatch) -> None:
    monkeypatch.setenv(
        "MCP_SERVERS",
        json.dumps(
            [
                {
                    "name": "server-a",
                    "transport": "streamable-http",
                    "url": "http://127.0.0.1:9000/mcp",
                    "verify_ssl": "false",
                }
            ]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="MCP_SERVERS contains an invalid server configuration",
    ):
        Settings.from_environment()


def test_tenant_policy_enforcement_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "TENANT_POLICY_ENFORCEMENT_ENABLED",
        raising=False,
    )
    monkeypatch.delenv(
        "TENANT_POLICIES",
        raising=False,
    )

    settings = Settings.from_environment()

    assert settings.tenant_policy_enforcement_enabled is False
    assert settings.tenant_policies == ()


def test_tenant_policies_are_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TENANT_POLICY_ENFORCEMENT_ENABLED",
        "true",
    )
    monkeypatch.setenv(
        "TENANT_POLICIES",
        json.dumps(
            [
                {
                    "tenant_id": "tenant-acme",
                    "allowed_tools": [
                        "rag.search",
                        "vehicle.data.query",
                    ],
                    "blocked_tools": [],
                    "allowed_mcp_servers": [
                        "document-server",
                    ],
                    "max_tokens_per_run": 10000,
                    "allow_cross_tenant_data": False,
                }
            ]
        ),
    )

    settings = Settings.from_environment()

    assert settings.tenant_policy_enforcement_enabled is True
    assert len(settings.tenant_policies) == 1

    policy = settings.tenant_policies[0]

    assert policy.tenant_id == "tenant-acme"
    assert policy.allowed_tools == frozenset(
        {
            "rag.search",
            "vehicle.data.query",
        }
    )
    assert policy.blocked_tools == frozenset()
    assert policy.allowed_mcp_servers == frozenset({"document-server"})
    assert policy.max_tokens_per_run == 10000
    assert policy.allow_cross_tenant_data is False


def test_tenant_policies_parse_model_governance_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TENANT_POLICY_ENFORCEMENT_ENABLED",
        "true",
    )
    monkeypatch.setenv(
        "TENANT_POLICIES",
        json.dumps(
            [
                {
                    "tenant_id": "tenant-acme",
                    "allowed_tools": ["rag.search"],
                    "allowed_models": [
                        "gpt-5",
                        "claude-sonnet-4",
                    ],
                    "allowed_providers": [
                        "openai",
                        "anthropic",
                    ],
                    "policy_id": "enterprise-model-policy",
                    "policy_version": "v7",
                }
            ]
        ),
    )

    settings = Settings.from_environment()

    policy = settings.tenant_policies[0]

    assert policy.allowed_models == frozenset(
        {
            "gpt-5",
            "claude-sonnet-4",
        }
    )
    assert policy.allowed_providers == frozenset(
        {
            "openai",
            "anthropic",
        }
    )
    assert policy.policy_id == "enterprise-model-policy"
    assert policy.policy_version == "v7"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("allowed_models", {}),
        ("allowed_providers", "openai"),
    ],
)
def test_tenant_policies_reject_invalid_model_governance_fields(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    value: object,
) -> None:
    monkeypatch.setenv(
        "TENANT_POLICY_ENFORCEMENT_ENABLED",
        "true",
    )
    monkeypatch.setenv(
        "TENANT_POLICIES",
        json.dumps(
            [
                {
                    "tenant_id": "tenant-acme",
                    field_name: value,
                }
            ]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="TENANT_POLICIES contains an invalid policy configuration",
    ):
        Settings.from_environment()


def test_tenant_policies_preserve_explicit_empty_model_governance_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TENANT_POLICY_ENFORCEMENT_ENABLED",
        "true",
    )
    monkeypatch.setenv(
        "TENANT_POLICIES",
        json.dumps(
            [
                {
                    "tenant_id": "tenant-acme",
                    "allowed_models": [],
                    "allowed_providers": [],
                }
            ]
        ),
    )

    settings = Settings.from_environment()
    policy = settings.tenant_policies[0]

    assert policy.allowed_models == frozenset()
    assert policy.allowed_providers == frozenset()


def test_tenant_policy_enforcement_requires_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TENANT_POLICY_ENFORCEMENT_ENABLED",
        "true",
    )
    monkeypatch.delenv(
        "TENANT_POLICIES",
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="TENANT_POLICIES must contain at least one policy",
    ):
        Settings.from_environment()


def test_tenant_policies_reject_non_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TENANT_POLICIES",
        json.dumps({"tenant_id": "tenant-acme"}),
    )

    with pytest.raises(
        RuntimeError,
        match="TENANT_POLICIES must contain a JSON array",
    ):
        Settings.from_environment()


def test_tenant_policies_reject_invalid_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "TENANT_POLICIES",
        json.dumps(
            [
                {
                    "tenant_id": "tenant-acme",
                    "allowed_tools": "rag.search",
                }
            ]
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="TENANT_POLICIES contains an invalid policy configuration",
    ):
        Settings.from_environment()
