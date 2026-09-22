from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

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
        )
