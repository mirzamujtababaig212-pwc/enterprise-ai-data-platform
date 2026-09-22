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
