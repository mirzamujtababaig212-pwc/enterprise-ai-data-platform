from __future__ import annotations

import subprocess
import sys


def _run_fresh_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_tools_package_import_does_not_trigger_agent_import_cycle() -> None:
    result = _run_fresh_python(
        """
import tools
from tools.mcp.config import MCPServerConfig
from tools.mcp.discovery import MCPToolDiscoveryService

print("import-boundary-ok")
"""
    )

    assert result.returncode == 0, result.stderr
    assert "import-boundary-ok" in result.stdout


def test_agent_modules_remain_importable_after_lightweight_package_init() -> None:
    result = _run_fresh_python(
        """
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.agents.tool_calls import AgentToolCall, AgentToolResult

print("agent-imports-ok")
"""
    )

    assert result.returncode == 0, result.stderr
    assert "agent-imports-ok" in result.stdout
