from __future__ import annotations

from datetime import UTC, datetime

from app.control_plane.mcp_servers.api_mapping import (
    config_request_to_domain,
    domain_to_lifecycle_response,
    update_payload_to_domain_values,
)
from app.control_plane.mcp_servers.models import (
    MCPServer,
    MCPServerDesiredState,
)
from app.control_plane.schemas.mcp import (
    MCPServerConfigRequest,
    MCPServerUpdateRequest,
)
from tools.mcp.config import MCPServerConfig, MCPToolCapability
from tools.mcp.recovery import MCPRecoveryPolicy


def make_server() -> MCPServer:
    now = datetime.now(UTC)

    config = MCPServerConfig(
        name="documents",
        transport="stdio",
        command="python",
        args=("server.py",),
        cwd="/opt/mcp",
        timeout=12.0,
        read_timeout=90.0,
        health_check_timeout=4.0,
        verify_ssl=False,
        recovery_policy=MCPRecoveryPolicy(
            max_attempts=5,
            initial_backoff=2.0,
            max_backoff=20.0,
            cooldown=45.0,
        ),
        tool_capabilities={
            "documents.search": MCPToolCapability(
                capability="search",
                risk_tier="low",
                permission_scope="documents:read",
            ),
        },
    )

    return MCPServer(
        server_id="server-1",
        tenant_id="tenant-a",
        name="documents",
        desired_state=MCPServerDesiredState.ACTIVE,
        config=config,
        secret_references={
            "authorization": "secret://mcp/documents/auth",
        },
        created_at=now,
        updated_at=now,
    )


def test_config_request_maps_to_typed_domain_config() -> None:
    payload = MCPServerConfigRequest(
        name="documents",
        transport="stdio",
        command="python",
        args=["server.py"],
        recovery_policy={
            "max_attempts": 4,
            "initial_backoff": 2,
            "max_backoff": 16,
            "cooldown": 30,
        },
        tool_capabilities={
            "documents.search": {
                "capability": "search",
                "risk_tier": "low",
                "side_effect": False,
                "permission_scope": "documents:read",
            }
        },
    )

    config = config_request_to_domain(payload)

    assert config.name == "documents"
    assert config.args == ("server.py",)
    assert config.recovery_policy.max_attempts == 4
    assert config.recovery_policy.initial_backoff == 2.0
    assert config.tool_capabilities["documents.search"].capability == "search"
    assert config.env == {}
    assert config.headers == {}


def test_domain_server_maps_to_safe_response() -> None:
    response = domain_to_lifecycle_response(make_server())

    assert response.server_id == "server-1"
    assert response.tenant_id == "tenant-a"
    assert response.name == "documents"
    assert response.desired_state == "active"
    assert response.args == ["server.py"]
    assert response.secret_references == {
        "authorization": "secret://mcp/documents/auth",
    }


def test_update_payload_preserves_omitted_values() -> None:
    server = make_server()

    payload = MCPServerUpdateRequest(
        timeout=25.0,
        secret_references={
            "authorization": "secret://mcp/documents/new-auth",
        },
    )

    config, secret_references = update_payload_to_domain_values(
        payload,
        server,
    )

    assert config.name == "documents"
    assert config.transport == "stdio"
    assert config.command == "python"
    assert config.args == ("server.py",)
    assert config.timeout == 25.0
    assert config.read_timeout == 90.0
    assert config.recovery_policy.max_attempts == 5
    assert secret_references == {
        "authorization": "secret://mcp/documents/new-auth",
    }


def test_api_models_reject_raw_secret_fields() -> None:
    payload = MCPServerConfigRequest(
        name="documents",
        transport="stdio",
        command="python",
    )

    try:
        payload.model_validate(
            {
                **payload.model_dump(),
                "env": {"TOKEN": "secret"},
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("raw env field was accepted")


def test_update_model_rejects_name_and_transport() -> None:
    payload = MCPServerUpdateRequest()

    try:
        payload.model_validate(
            {
                **payload.model_dump(),
                "name": "renamed",
                "transport": "streamable-http",
            }
        )
    except ValueError:
        pass
    else:
        raise AssertionError("immutable MCP fields were accepted")
