import asyncio

import pytest

from tools.authorization.in_memory import (
    InMemoryToolAuthorizer,
)
from tools.authorization.service import (
    ToolAuthorizationService,
)
from tools.execution.service import ToolExecutionService
from tools.models import ToolDefinition
from tools.registry.in_memory import InMemoryToolRegistry


class FakeTool:
    def __init__(
        self,
        name: str = "test_tool",
        enabled: bool = True,
    ):
        self._definition = ToolDefinition(
            name=name,
            description="A test tool.",
            enabled=enabled,
        )

        self.execution_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execution_count += 1

        return {
            "status": "success",
            "arguments": arguments,
        }


class FailingTool:
    def __init__(self):
        self._definition = ToolDefinition(
            name="failing_tool",
            description="A failing test tool.",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        raise RuntimeError("simulated tool failure")


class SlowTool:
    def __init__(self):
        self._definition = ToolDefinition(
            name="slow_tool",
            description="A slow test tool.",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        await asyncio.sleep(0.2)
        return {"status": "completed"}


@pytest.mark.asyncio
async def test_execute_calls_registered_tool():
    registry = InMemoryToolRegistry()
    tool = FakeTool()

    await registry.register(tool)

    service = ToolExecutionService(registry)

    result = await service.execute(
        "test_tool",
        {"value": 42},
    )

    assert result.success is True
    assert result.tool_name == "test_tool"
    assert result.output == {
        "status": "success",
        "arguments": {"value": 42},
    }
    assert result.error is None
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_execute_unknown_tool_returns_failure():
    registry = InMemoryToolRegistry()
    service = ToolExecutionService(registry)

    result = await service.execute(
        "missing_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "missing_tool"
    assert result.output is None
    assert result.error == "Tool not found: missing_tool"


@pytest.mark.asyncio
async def test_execute_disabled_tool_returns_failure():
    registry = InMemoryToolRegistry()

    await registry.register(
        FakeTool(
            name="disabled_tool",
            enabled=False,
        )
    )

    service = ToolExecutionService(registry)

    result = await service.execute(
        "disabled_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "disabled_tool"
    assert result.error == "Tool is disabled: disabled_tool"


@pytest.mark.asyncio
async def test_execute_empty_tool_name_is_rejected():
    registry = InMemoryToolRegistry()
    service = ToolExecutionService(registry)

    with pytest.raises(
        ValueError,
        match="Tool name must not be empty",
    ):
        await service.execute(
            "",
            {},
        )


@pytest.mark.asyncio
async def test_execute_tool_failure_is_isolated():
    registry = InMemoryToolRegistry()

    await registry.register(FailingTool())

    service = ToolExecutionService(registry)

    result = await service.execute(
        "failing_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "failing_tool"
    assert result.output is None
    assert result.error == "RuntimeError: simulated tool failure"


@pytest.mark.asyncio
async def test_execute_timeout_is_handled():
    registry = InMemoryToolRegistry()

    await registry.register(SlowTool())

    service = ToolExecutionService(
        registry,
        default_timeout_seconds=0.05,
    )

    result = await service.execute(
        "slow_tool",
        {},
    )

    assert result.success is False
    assert result.tool_name == "slow_tool"
    assert result.output is None
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_execute_custom_timeout_is_used():
    registry = InMemoryToolRegistry()

    await registry.register(SlowTool())

    service = ToolExecutionService(
        registry,
        default_timeout_seconds=1.0,
    )

    result = await service.execute(
        "slow_tool",
        {},
        timeout_seconds=0.05,
    )

    assert result.success is False
    assert "timed out" in result.error


def test_invalid_default_timeout_is_rejected():
    registry = InMemoryToolRegistry()

    with pytest.raises(
        ValueError,
        match="default_timeout_seconds must be greater than zero",
    ):
        ToolExecutionService(
            registry,
            default_timeout_seconds=0,
        )


@pytest.mark.asyncio
async def test_invalid_custom_timeout_is_rejected():
    registry = InMemoryToolRegistry()

    service = ToolExecutionService(registry)

    with pytest.raises(
        ValueError,
        match="timeout_seconds must be greater than zero",
    ):
        await service.execute(
            "test_tool",
            {},
            timeout_seconds=0,
        )


@pytest.mark.asyncio
async def test_authorized_principal_can_execute_tool():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = FakeTool()

    await registry.register(tool)

    await authorizer.allow(
        "agent:research",
        "test_tool",
    )

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {"value": 42},
        principal="agent:research",
    )

    assert result.success is True
    assert result.output == {
        "status": "success",
        "arguments": {"value": 42},
    }
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_unauthorized_principal_cannot_execute_tool():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = FakeTool()

    await registry.register(tool)

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {},
        principal="agent:restricted",
    )

    assert result.success is False
    assert result.tool_name == "test_tool"
    assert result.error == "Tool is not authorized for this principal."
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_authorization_requires_principal():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    await registry.register(FakeTool())

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {},
    )

    assert result.success is False
    assert result.error == ("Principal is required when " "tool authorization is enabled.")


@pytest.mark.asyncio
async def test_authorization_happens_before_tool_execution():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = FakeTool()

    await registry.register(tool)

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
    )

    result = await service.execute(
        "test_tool",
        {},
        principal="agent:restricted",
    )

    assert result.success is False
    assert tool.execution_count == 0


@pytest.mark.asyncio
async def test_authorized_tool_still_respects_timeout():
    registry = InMemoryToolRegistry()
    authorizer = InMemoryToolAuthorizer()

    tool = SlowTool()

    await registry.register(tool)

    await authorizer.allow(
        "agent:research",
        "slow_tool",
    )

    authorization_service = ToolAuthorizationService(authorizer)

    service = ToolExecutionService(
        registry,
        authorization_service=authorization_service,
        default_timeout_seconds=0.05,
    )

    result = await service.execute(
        "slow_tool",
        {},
        principal="agent:research",
    )

    assert result.success is False
    assert "timed out" in result.error
