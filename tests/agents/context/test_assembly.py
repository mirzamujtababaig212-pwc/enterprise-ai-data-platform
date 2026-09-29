from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ai_platform.agents.context.assembly import AgentContextAssembly
from ai_platform.agents.lifecycle import AgentExecutionLifecycleState
from ai_platform.agents.llm_context import AgentLLMContext, UnavailableLLMGateway
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.llm_messages import AgentMessage
from ai_platform.agents.tool_context import AgentToolContext
from memory.context.builder import MemoryContext
from memory.models import MemoryItem


class EmptyToolRegistry:
    async def get(self, name: str):
        return None

    async def list_tools(self):
        return []


def _components():
    definition = AgentDefinition(
        name="context-test-agent",
        description="Context assembly test agent",
        system_prompt="You are a test agent.",
    )

    tools = AgentToolContext(
        EmptyToolRegistry(),
        definition,
    )

    llm = AgentLLMContext(
        UnavailableLLMGateway(),
        definition.llm_config,
    )

    return definition, tools, llm


def test_assembly_preserves_runtime_context_inputs():
    _, tools, llm = _components()

    request = AgentRequest(
        input="hello",
        session_id="session-1",
        user_id="user-1",
        tenant_id="tenant-1",
    )

    ownership_lost = asyncio.Event()
    cancellation_requested = asyncio.Event()
    lifecycle_state = AgentExecutionLifecycleState()

    context = AgentContextAssembly.assemble(
        request,
        tools=tools,
        llm=llm,
        history=(),
        memory=None,
        run_id="run-1",
        lease_id="lease-1",
        execution_ownership_lost=ownership_lost,
        cancellation_requested=cancellation_requested,
        lifecycle_state=lifecycle_state,
    )

    assert context.request is request
    assert context.tools is tools
    assert context.llm is llm
    assert context.history == ()
    assert context.memory is None
    assert context.run_id == "run-1"
    assert context.lease_id == "lease-1"
    assert context.execution_ownership_lost is ownership_lost
    assert context.cancellation_requested is cancellation_requested
    assert context.lifecycle_state is lifecycle_state


def test_assembly_preserves_history_memory_and_optional_runtime_providers():
    _, tools, llm = _components()

    request = AgentRequest(input="continue")
    history = (
        AgentMessage(role="user", content="previous question"),
        AgentMessage(role="assistant", content="previous answer"),
    )
    memory_item = MemoryItem(
        id="memory-1",
        memory_type="semantic",
        content="remembered context",
        namespace="context-test",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    memory = MemoryContext(
        working=(),
        semantic=(memory_item,),
        episodic=(),
    )
    lifecycle_state = AgentExecutionLifecycleState()

    context = AgentContextAssembly.assemble(
        request,
        tools=tools,
        llm=llm,
        history=history,
        memory=memory,
        lifecycle_state=lifecycle_state,
    )

    assert context.history is history
    assert context.memory is memory
    assert context.lifecycle_state is lifecycle_state
