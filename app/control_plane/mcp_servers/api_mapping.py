from __future__ import annotations

from typing import Any

from app.control_plane.mcp_servers.models import MCPServer
from app.control_plane.schemas.mcp import (
    MCPServerConfigRequest,
    MCPServerLifecycleResponse,
)
from tools.mcp.config import MCPServerConfig, MCPToolCapability
from tools.mcp.recovery import MCPRecoveryPolicy


def config_request_to_domain(
    payload: MCPServerConfigRequest,
) -> MCPServerConfig:
    recovery = payload.recovery_policy

    recovery_policy = MCPRecoveryPolicy(
        max_attempts=int(recovery.get("max_attempts", 3)),
        initial_backoff=float(recovery.get("initial_backoff", 1.0)),
        max_backoff=float(recovery.get("max_backoff", 30.0)),
        cooldown=float(recovery.get("cooldown", 60.0)),
    )

    capabilities = {
        tool_name: MCPToolCapability(
            capability=capability_data["capability"],
            risk_tier=capability_data.get("risk_tier", "low"),
            side_effect=bool(capability_data.get("side_effect", False)),
            permission_scope=capability_data.get("permission_scope"),
        )
        for tool_name, capability_data in payload.tool_capabilities.items()
    }

    return MCPServerConfig(
        name=payload.name,
        transport=payload.transport,
        command=payload.command,
        args=tuple(payload.args),
        cwd=payload.cwd,
        url=payload.url,
        timeout=payload.timeout,
        read_timeout=payload.read_timeout,
        health_check_timeout=payload.health_check_timeout,
        verify_ssl=payload.verify_ssl,
        recovery_policy=recovery_policy,
        tool_capabilities=capabilities,
    )


def domain_to_lifecycle_response(
    server: MCPServer,
) -> MCPServerLifecycleResponse:
    config = server.config

    return MCPServerLifecycleResponse(
        server_id=server.server_id,
        tenant_id=server.tenant_id,
        name=server.name,
        transport=config.transport,
        desired_state=server.desired_state.value,
        command=config.command,
        args=list(config.args),
        cwd=config.cwd,
        url=config.url,
        timeout=config.timeout,
        read_timeout=config.read_timeout,
        health_check_timeout=config.health_check_timeout,
        verify_ssl=config.verify_ssl,
        recovery_policy={
            "max_attempts": config.recovery_policy.max_attempts,
            "initial_backoff": config.recovery_policy.initial_backoff,
            "max_backoff": config.recovery_policy.max_backoff,
            "cooldown": config.recovery_policy.cooldown,
        },
        tool_capabilities={
            tool_name: {
                "capability": capability.capability,
                "risk_tier": capability.risk_tier,
                "side_effect": capability.side_effect,
                "permission_scope": capability.permission_scope,
            }
            for tool_name, capability in config.tool_capabilities.items()
        },
        secret_references=dict(server.secret_references),
        created_at=server.created_at,
        updated_at=server.updated_at,
    )


def update_payload_to_domain_values(
    payload: Any,
    existing: MCPServer,
) -> tuple[MCPServerConfig, dict[str, str]]:
    config = existing.config

    recovery_data = (
        payload.recovery_policy
        if payload.recovery_policy is not None
        else {
            "max_attempts": config.recovery_policy.max_attempts,
            "initial_backoff": config.recovery_policy.initial_backoff,
            "max_backoff": config.recovery_policy.max_backoff,
            "cooldown": config.recovery_policy.cooldown,
        }
    )

    capability_data = (
        payload.tool_capabilities
        if payload.tool_capabilities is not None
        else {
            tool_name: {
                "capability": capability.capability,
                "risk_tier": capability.risk_tier,
                "side_effect": capability.side_effect,
                "permission_scope": capability.permission_scope,
            }
            for tool_name, capability in config.tool_capabilities.items()
        }
    )

    updated_config = MCPServerConfig(
        name=existing.name,
        transport=config.transport,
        command=payload.command if payload.command is not None else config.command,
        args=(tuple(payload.args) if payload.args is not None else config.args),
        cwd=payload.cwd if payload.cwd is not None else config.cwd,
        url=payload.url if payload.url is not None else config.url,
        timeout=payload.timeout if payload.timeout is not None else config.timeout,
        read_timeout=(
            payload.read_timeout if payload.read_timeout is not None else config.read_timeout
        ),
        health_check_timeout=(
            payload.health_check_timeout
            if payload.health_check_timeout is not None
            else config.health_check_timeout
        ),
        verify_ssl=(payload.verify_ssl if payload.verify_ssl is not None else config.verify_ssl),
        recovery_policy=MCPRecoveryPolicy(
            max_attempts=int(recovery_data.get("max_attempts", 3)),
            initial_backoff=float(recovery_data.get("initial_backoff", 1.0)),
            max_backoff=float(recovery_data.get("max_backoff", 30.0)),
            cooldown=float(recovery_data.get("cooldown", 60.0)),
        ),
        tool_capabilities={
            tool_name: MCPToolCapability(
                capability=capability["capability"],
                risk_tier=capability.get("risk_tier", "low"),
                side_effect=bool(capability.get("side_effect", False)),
                permission_scope=capability.get("permission_scope"),
            )
            for tool_name, capability in capability_data.items()
        },
    )

    secret_references = (
        dict(payload.secret_references)
        if payload.secret_references is not None
        else dict(existing.secret_references)
    )

    return updated_config, secret_references
