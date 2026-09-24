from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from ai_platform.agents.policy import TenantPolicy
from tools.mcp.config import MCPServerConfig


@dataclass(frozen=True)
class Settings:
    environment: str
    aws_region: str
    default_provider: str
    log_level: str
    provider_credentials: dict[str, Any]
    external_evaluation_release_required: bool
    agent_run_lease_duration_seconds: int
    agent_run_max_recovery_attempts: int
    mcp_servers: tuple[MCPServerConfig, ...] = ()
    tenant_policy_enforcement_enabled: bool = False
    tenant_policies: tuple[TenantPolicy, ...] = ()

    @classmethod
    def from_environment(cls) -> "Settings":
        credentials_raw = os.getenv("PROVIDER_CREDENTIALS", "{}")

        try:
            credentials = json.loads(credentials_raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("PROVIDER_CREDENTIALS must contain valid JSON") from exc

        mcp_servers_raw = os.getenv("MCP_SERVERS", "").strip()

        if not mcp_servers_raw:
            mcp_servers: tuple[MCPServerConfig, ...] = ()
        else:
            try:
                mcp_servers_payload = json.loads(mcp_servers_raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError("MCP_SERVERS must contain valid JSON") from exc

            if not isinstance(mcp_servers_payload, list):
                raise RuntimeError("MCP_SERVERS must contain a JSON array.")

            try:
                parsed_servers: list[MCPServerConfig] = []

                for server in mcp_servers_payload:
                    if not isinstance(server, dict):
                        raise ValueError("MCP server entry must be an object.")

                    args = server.get("args", ())
                    env = server.get("env", {})
                    headers = server.get("headers", {})
                    verify_ssl = server.get("verify_ssl", True)

                    if not isinstance(args, (list, tuple)):
                        raise ValueError("MCP server args must be an array.")

                    if not isinstance(env, dict):
                        raise ValueError("MCP server env must be an object.")

                    if not isinstance(headers, dict):
                        raise ValueError("MCP server headers must be an object.")

                    if not isinstance(verify_ssl, bool):
                        raise ValueError("MCP server verify_ssl must be a boolean.")

                    parsed_servers.append(
                        MCPServerConfig(
                            name=server["name"],
                            transport=server["transport"],
                            command=server.get("command"),
                            args=tuple(args),
                            env=dict(env),
                            cwd=server.get("cwd"),
                            url=server.get("url"),
                            headers=dict(headers),
                            timeout=float(server.get("timeout", 30.0)),
                            read_timeout=float(server.get("read_timeout", 300.0)),
                            verify_ssl=verify_ssl,
                        )
                    )

                mcp_servers = tuple(parsed_servers)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("MCP_SERVERS contains an invalid server configuration.") from exc

        tenant_policy_enforcement_enabled = os.getenv(
            "TENANT_POLICY_ENFORCEMENT_ENABLED",
            "false",
        ).strip().lower() in {"1", "true", "yes", "on"}

        tenant_policies_raw = os.getenv("TENANT_POLICIES", "").strip()

        if not tenant_policies_raw:
            tenant_policies: tuple[TenantPolicy, ...] = ()
        else:
            try:
                tenant_policies_payload = json.loads(tenant_policies_raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError("TENANT_POLICIES must contain valid JSON") from exc

            if not isinstance(tenant_policies_payload, list):
                raise RuntimeError("TENANT_POLICIES must contain a JSON array.")

            try:
                parsed_policies: list[TenantPolicy] = []

                for policy in tenant_policies_payload:
                    if not isinstance(policy, dict):
                        raise ValueError("Tenant policy entry must be an object.")

                    tenant_id = policy["tenant_id"]

                    allowed_tools = policy.get("allowed_tools", [])
                    blocked_tools = policy.get("blocked_tools", [])
                    allowed_mcp_servers = policy.get(
                        "allowed_mcp_servers",
                        [],
                    )
                    allowed_models = policy.get("allowed_models")
                    allowed_providers = policy.get("allowed_providers")

                    if not isinstance(allowed_tools, list):
                        raise ValueError("Tenant policy allowed_tools must be an array.")

                    if not isinstance(blocked_tools, list):
                        raise ValueError("Tenant policy blocked_tools must be an array.")

                    if not isinstance(allowed_mcp_servers, list):
                        raise ValueError("Tenant policy allowed_mcp_servers must be an array.")

                    if allowed_models is not None and not isinstance(
                        allowed_models,
                        list,
                    ):
                        raise ValueError("Tenant policy allowed_models must be an array.")

                    if allowed_providers is not None and not isinstance(
                        allowed_providers,
                        list,
                    ):
                        raise ValueError("Tenant policy allowed_providers must be an array.")

                    max_tokens_per_run = policy.get("max_tokens_per_run")

                    if max_tokens_per_run is not None and not isinstance(
                        max_tokens_per_run,
                        int,
                    ):
                        raise ValueError("Tenant policy max_tokens_per_run must be an integer.")

                    allow_cross_tenant_data = policy.get(
                        "allow_cross_tenant_data",
                        False,
                    )

                    if not isinstance(allow_cross_tenant_data, bool):
                        raise ValueError(
                            "Tenant policy allow_cross_tenant_data " "must be a boolean."
                        )

                    parsed_policies.append(
                        TenantPolicy(
                            tenant_id=tenant_id,
                            allowed_tools=frozenset(allowed_tools),
                            blocked_tools=frozenset(blocked_tools),
                            allowed_mcp_servers=frozenset(allowed_mcp_servers),
                            max_tokens_per_run=max_tokens_per_run,
                            allow_cross_tenant_data=allow_cross_tenant_data,
                            allowed_models=(
                                None if allowed_models is None else frozenset(allowed_models)
                            ),
                            allowed_providers=(
                                None if allowed_providers is None else frozenset(allowed_providers)
                            ),
                            policy_id=policy.get("policy_id"),
                            policy_version=policy.get("policy_version"),
                        )
                    )

                tenant_policies = tuple(parsed_policies)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "TENANT_POLICIES contains an invalid policy configuration."
                ) from exc

        if tenant_policy_enforcement_enabled and not tenant_policies:
            raise RuntimeError(
                "TENANT_POLICIES must contain at least one policy when "
                "tenant policy enforcement is enabled."
            )

        return cls(
            environment=os.getenv("ENVIRONMENT", "dev"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            default_provider=os.getenv(
                "DEFAULT_PROVIDER",
                "openai",
            ).lower(),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            provider_credentials=credentials,
            external_evaluation_release_required=(
                os.getenv(
                    "EXTERNAL_EVALUATION_RELEASE_REQUIRED",
                    "false",
                )
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            ),
            agent_run_lease_duration_seconds=int(
                os.getenv("AGENT_RUN_LEASE_DURATION_SECONDS", "60")
            ),
            agent_run_max_recovery_attempts=int(os.getenv("AGENT_RUN_MAX_RECOVERY_ATTEMPTS", "3")),
            mcp_servers=mcp_servers,
            tenant_policy_enforcement_enabled=tenant_policy_enforcement_enabled,
            tenant_policies=tenant_policies,
        )
