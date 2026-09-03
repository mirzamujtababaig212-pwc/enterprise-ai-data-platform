from __future__ import annotations

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Enterprise AI Platform Policy Server")


@mcp.tool()
def get_policy(policy_name: str) -> dict:
    """
    Return a deterministic enterprise policy.
    """
    policies = {
        "security": {
            "name": "security",
            "classification": "enterprise",
            "require_mfa": True,
            "audit_logging": True,
        },
        "data-retention": {
            "name": "data-retention",
            "classification": "enterprise",
            "retention_days": 365,
        },
    }

    policy = policies.get(policy_name)

    if policy is None:
        return {
            "policy_name": policy_name,
            "found": False,
        }

    return {
        "policy_name": policy_name,
        "found": True,
        "policy": policy,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
