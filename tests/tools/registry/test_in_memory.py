import pytest

from tools.models import ToolDefinition
from tools.registry.in_memory import InMemoryToolRegistry


class FakeTool:
    def __init__(
        self,
        name: str = "test_tool",
        enabled: bool = True,
    ) -> None:
        self._definition = ToolDefinition(
            name=name,
            description="A test tool.",
            input_schema={"type": "object"},
            enabled=enabled,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        return {"received": arguments}


@pytest.mark.asyncio
async def test_register_and_get_tool():
    registry = InMemoryToolRegistry()
    tool = FakeTool()

    await registry.register(tool)

    result = await registry.get("test_tool")

    assert result is tool


@pytest.mark.asyncio
async def test_list_tools_returns_enabled_definitions():
    registry = InMemoryToolRegistry()

    await registry.register(FakeTool("tool_a"))
    await registry.register(FakeTool("tool_b"))

    tools = await registry.list_tools()

    assert len(tools) == 2
    assert {tool.name for tool in tools} == {"tool_a", "tool_b"}


@pytest.mark.asyncio
async def test_disabled_tool_is_not_listed():
    registry = InMemoryToolRegistry()

    await registry.register(FakeTool("enabled_tool", enabled=True))
    await registry.register(FakeTool("disabled_tool", enabled=False))

    tools = await registry.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "enabled_tool"


@pytest.mark.asyncio
async def test_remove_tool():
    registry = InMemoryToolRegistry()
    tool = FakeTool()

    await registry.register(tool)
    await registry.remove("test_tool")

    result = await registry.get("test_tool")

    assert result is None


@pytest.mark.asyncio
async def test_empty_tool_name_is_rejected():
    registry = InMemoryToolRegistry()

    with pytest.raises(
        ValueError,
        match="Tool name must not be empty",
    ):
        await registry.get("")
