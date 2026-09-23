from __future__ import annotations

from typing import Any

from tools.mcp.config import MCPServerConfig, MCPToolCapability
from tools.mcp.recovery import MCPRecoveryPolicy


def serialize_mcp_server_config(config: MCPServerConfig) -> dict[str, Any]:
    """
    Serialize durable, non-secret MCP server configuration.

    Environment variables and HTTP headers may contain credentials or other
    secrets and therefore must never be persisted in the MCP server record.
    Callers must persist secret references separately.
    """
    if config.env:
        raise ValueError(
            "MCP server environment values must not be persisted directly; "
            "use secret references instead."
        )

    if config.headers:
        raise ValueError(
            "MCP server HTTP headers must not be persisted directly; "
            "use secret references instead."
        )

    return {
        "name": config.name,
        "transport": config.transport,
        "command": config.command,
        "args": list(config.args),
        "cwd": config.cwd,
        "url": config.url,
        "timeout": config.timeout,
        "read_timeout": config.read_timeout,
        "health_check_timeout": config.health_check_timeout,
        "verify_ssl": config.verify_ssl,
        "recovery_policy": {
            "max_attempts": config.recovery_policy.max_attempts,
            "initial_backoff": config.recovery_policy.initial_backoff,
            "max_backoff": config.recovery_policy.max_backoff,
            "cooldown": config.recovery_policy.cooldown,
        },
        "tool_capabilities": {
            tool_name: {
                "capability": capability.capability,
                "risk_tier": capability.risk_tier,
                "side_effect": capability.side_effect,
                "permission_scope": capability.permission_scope,
            }
            for tool_name, capability in config.tool_capabilities.items()
        },
    }


def deserialize_mcp_server_config(data: dict[str, Any]) -> MCPServerConfig:
    """
    Reconstruct MCPServerConfig from the durable non-secret representation.
    """
    recovery_data = data.get("recovery_policy", {})
    recovery_policy = MCPRecoveryPolicy(
        max_attempts=int(recovery_data.get("max_attempts", 3)),
        initial_backoff=float(recovery_data.get("initial_backoff", 1.0)),
        max_backoff=float(recovery_data.get("max_backoff", 30.0)),
        cooldown=float(recovery_data.get("cooldown", 60.0)),
    )

    capabilities = {
        tool_name: MCPToolCapability(
            capability=capability_data["capability"],
            risk_tier=capability_data.get("risk_tier", "low"),
            side_effect=bool(capability_data.get("side_effect", False)),
            permission_scope=capability_data.get("permission_scope"),
        )
        for tool_name, capability_data in data.get("tool_capabilities", {}).items()
    }

    return MCPServerConfig(
        name=data["name"],
        transport=data["transport"],
        command=data.get("command"),
        args=tuple(data.get("args", ())),
        cwd=data.get("cwd"),
        url=data.get("url"),
        timeout=float(data.get("timeout", 30.0)),
        read_timeout=float(data.get("read_timeout", 300.0)),
        health_check_timeout=float(data.get("health_check_timeout", 5.0)),
        verify_ssl=bool(data.get("verify_ssl", True)),
        recovery_policy=recovery_policy,
        tool_capabilities=capabilities,
    )
