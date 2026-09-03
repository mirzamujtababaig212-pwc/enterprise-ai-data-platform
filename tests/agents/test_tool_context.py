from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ai_platform.agents.models import AgentDefinition
from ai_platform.agents.tool_context import AgentToolContext
from tools.models import ToolDefinition
from tools.registry.in_memory import InMemoryToolRegistry


@dataclass
class FakeTool:
    definition: ToolDefinition
    output: Any = None

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> Any:
        return self.output


def make_agent_definition(
    *,
    tool_names: tuple[str, ...] = (),
) -> AgentDefinition:
    return AgentDefinition(
        name="test-agent",
        description="Test agent",
        system_prompt="You are a test agent.",
        tool_names=tool_names,
    )


def make_tool(
    name: str,
    *,
    enabled: bool = True,
    output: Any = None,
) -> FakeTool:
    return FakeTool(
        definition=ToolDefinition(
            name=name,
            description=f"{name} tool",
            enabled=enabled,
        ),
        output=output,
    )


@pytest.mark.asyncio
async def test_agent_name_matches_definition() -> None:
    registry = InMemoryToolRegistry()

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
    )

    assert context.agent_name == "test-agent"


@pytest.mark.asyncio
async def test_tool_names_match_agent_definition() -> None:
    registry = InMemoryToolRegistry()

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search", "calculator"),
        ),
    )

    assert context.tool_names == (
        "search",
        "calculator",
    )


@pytest.mark.asyncio
async def test_get_tool_returns_declared_registered_tool() -> None:
    registry = InMemoryToolRegistry()

    tool = make_tool(
        "search",
        output={"results": []},
    )

    await registry.register(tool)

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
    )

    resolved = await context.get_tool("search")

    assert resolved is tool


@pytest.mark.asyncio
async def test_get_tool_returns_none_when_declared_tool_is_not_registered() -> None:
    registry = InMemoryToolRegistry()

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
    )

    resolved = await context.get_tool("search")

    assert resolved is None


@pytest.mark.asyncio
async def test_get_tool_rejects_undeclared_tool() -> None:
    registry = InMemoryToolRegistry()

    tool = make_tool("calculator")

    await registry.register(tool)

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="not declared for agent",
    ):
        await context.get_tool("calculator")


@pytest.mark.asyncio
async def test_get_tool_rejects_empty_name() -> None:
    registry = InMemoryToolRegistry()

    context = AgentToolContext(
        registry,
        make_agent_definition(),
    )

    with pytest.raises(
        ValueError,
        match="Tool name must not be empty",
    ):
        await context.get_tool("")


@pytest.mark.asyncio
async def test_list_tools_returns_only_agent_declared_tools() -> None:
    registry = InMemoryToolRegistry()

    await registry.register(
        make_tool("search"),
    )

    await registry.register(
        make_tool("calculator"),
    )

    await registry.register(
        make_tool("weather"),
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search", "weather"),
        ),
    )

    definitions = await context.list_tools()

    assert [definition.name for definition in definitions] == [
        "search",
        "weather",
    ]


@pytest.mark.asyncio
async def test_list_tools_excludes_disabled_tools() -> None:
    registry = InMemoryToolRegistry()

    await registry.register(
        make_tool("search"),
    )

    await registry.register(
        make_tool(
            "weather",
            enabled=False,
        ),
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search", "weather"),
        ),
    )

    definitions = await context.list_tools()

    assert [definition.name for definition in definitions] == [
        "search",
    ]


@pytest.mark.asyncio
async def test_list_tools_returns_empty_when_agent_has_no_tools() -> None:
    registry = InMemoryToolRegistry()

    await registry.register(
        make_tool("search"),
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(),
    )

    assert await context.list_tools() == []


@pytest.mark.asyncio
async def test_execute_delegates_to_registered_tool() -> None:
    registry = InMemoryToolRegistry()

    tool = make_tool(
        "search",
        output={"status": "ok"},
    )

    await registry.register(tool)

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
    )

    result = await context.execute(
        "search",
        {"query": "hello"},
    )

    assert result.tool_name == "search"
    assert result.success is True
    assert result.output == {"status": "ok"}
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_returns_failed_result_when_tool_is_missing() -> None:
    registry = InMemoryToolRegistry()

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
    )

    result = await context.execute(
        "search",
        {},
    )

    assert result.tool_name == "search"
    assert result.success is False
    assert result.output is None
    assert result.error == "Tool not found: search"


@pytest.mark.asyncio
async def test_execute_returns_failed_result_when_tool_is_disabled() -> None:
    registry = InMemoryToolRegistry()

    await registry.register(
        make_tool(
            "search",
            enabled=False,
        ),
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
    )

    result = await context.execute(
        "search",
        {},
    )

    assert result.tool_name == "search"
    assert result.success is False
    assert result.output is None
    assert result.error == "Tool is disabled: search"


@pytest.mark.asyncio
async def test_execute_rejects_undeclared_tool() -> None:
    registry = InMemoryToolRegistry()

    await registry.register(
        make_tool(
            "calculator",
            output=42,
        ),
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
    )

    with pytest.raises(
        ValueError,
        match="not declared for agent",
    ):
        await context.execute(
            "calculator",
            {},
        )


class FakeExecutionService:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        principal: str | None = None,
        timeout_seconds: float | None = None,
    ) -> Any:
        self.calls.append(
            {
                "tool_name": tool_name,
                "arguments": arguments,
                "principal": principal,
                "timeout_seconds": timeout_seconds,
            }
        )

        return self.result


@pytest.mark.asyncio
async def test_execute_delegates_to_execution_service() -> None:
    registry = InMemoryToolRegistry()

    await registry.register(
        make_tool(
            "search",
            output={"direct": "tool output"},
        ),
    )

    execution_service = FakeExecutionService(
        {
            "success": True,
            "output": {"service": "result"},
        },
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
        execution_service=execution_service,
    )

    result = await context.execute(
        "search",
        {"query": "hello"},
    )

    assert result == {
        "success": True,
        "output": {"service": "result"},
    }

    assert execution_service.calls == [
        {
            "tool_name": "search",
            "arguments": {"query": "hello"},
            "principal": None,
            "timeout_seconds": None,
        }
    ]


@pytest.mark.asyncio
async def test_execute_passes_principal_to_execution_service() -> None:
    registry = InMemoryToolRegistry()

    await registry.register(
        make_tool("search"),
    )

    execution_service = FakeExecutionService(
        {"success": True},
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
        execution_service=execution_service,
    )

    await context.execute(
        "search",
        {},
        principal="user-123",
    )

    assert execution_service.calls[0]["principal"] == "user-123"


@pytest.mark.asyncio
async def test_execute_passes_timeout_to_execution_service() -> None:
    registry = InMemoryToolRegistry()

    await registry.register(
        make_tool("search"),
    )

    execution_service = FakeExecutionService(
        {"success": True},
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
        execution_service=execution_service,
    )

    await context.execute(
        "search",
        {},
        timeout_seconds=12.5,
    )

    assert execution_service.calls[0]["timeout_seconds"] == 12.5


@pytest.mark.asyncio
async def test_execute_rejects_undeclared_tool_before_execution_service() -> None:
    registry = InMemoryToolRegistry()

    execution_service = FakeExecutionService(
        {"success": True},
    )

    context = AgentToolContext(
        registry,
        make_agent_definition(
            tool_names=("search",),
        ),
        execution_service=execution_service,
    )

    with pytest.raises(
        ValueError,
        match="not declared for agent",
    ):
        await context.execute(
            "calculator",
            {},
        )

    assert execution_service.calls == []
