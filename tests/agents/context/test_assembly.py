from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from ai_platform.agents.context.assembly import ContextSource, AgentContextAssembly
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


def test_describe_context_preserves_messages_without_content_provenance():
    _, tools, llm = _components()

    request = AgentRequest(
        input="What happened?",
        session_id="session-1",
    )

    context = AgentContextAssembly.assemble(
        request,
        tools=tools,
        llm=llm,
    )

    messages = context.build_llm_messages()

    result = AgentContextAssembly.describe(
        context,
        messages,
    )

    assert result.messages == messages
    assert [source.source_type for source in result.sources] == [
        "system_prompt",
        "user_input",
    ]

    assert result.diagnostics == {
        "total_messages": len(messages),
        "source_counts": {
            "system_prompt": 1,
            "user_input": 1,
        },
    }

    assert all("content" not in source.provenance_metadata for source in result.sources)


def test_describe_context_tracks_history_and_tool_results_without_outputs():
    _, tools, llm = _components()

    request = AgentRequest(
        input="Continue.",
        session_id="session-1",
    )

    history = (
        AgentMessage(
            role="user",
            content="Previous question.",
        ),
        AgentMessage(
            role="assistant",
            content="Previous answer.",
        ),
    )

    context = AgentContextAssembly.assemble(
        request,
        tools=tools,
        llm=llm,
        history=history,
    )

    messages = context.build_llm_messages(
        tool_results=("vehicle result",),
    )

    result = AgentContextAssembly.describe(
        context,
        messages,
    )

    assert result.messages == messages

    source_types = [source.source_type for source in result.sources]

    assert source_types == [
        "system_prompt",
        "chat_history",
        "user_input",
        "tool_result",
    ]

    assert result.diagnostics["source_counts"] == {
        "system_prompt": 1,
        "chat_history": 1,
        "user_input": 1,
        "tool_result": 1,
    }

    serialized_metadata = repr(result.sources)

    assert "vehicle result" not in serialized_metadata
    assert "Previous question." not in serialized_metadata
    assert "Previous answer." not in serialized_metadata


def test_describe_context_preserves_memory_retrieval_provenance():
    from datetime import datetime, timezone

    from memory.models import MemoryItem
    from memory.context.builder import MemoryRetrievalResult

    _, tools, llm = _components()

    semantic_item = MemoryItem(
        id="semantic-1",
        memory_type="semantic",
        content="The platform uses Databricks.",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )

    episodic_item = MemoryItem(
        id="episodic-1",
        memory_type="episodic",
        content="The previous deployment completed successfully.",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )

    memory = MemoryContext(
        working=(),
        semantic=(semantic_item,),
        episodic=(episodic_item,),
        semantic_results=(
            MemoryRetrievalResult(
                item=semantic_item,
                retrieval_method="vector",
                rank=1,
                retrieval_score=0.91,
                reranker_score=0.87,
                provenance={"source": "qdrant"},
            ),
        ),
        episodic_results=(
            MemoryRetrievalResult(
                item=episodic_item,
                retrieval_method="hybrid",
                rank=2,
                retrieval_score=0.82,
                provenance={"source": "memory-index"},
            ),
        ),
    )

    context = AgentContextAssembly.assemble(
        AgentRequest(
            input="What do you remember?",
            memory_namespace="project-a",
        ),
        tools=tools,
        llm=llm,
        memory=memory,
    )

    messages = context.build_llm_messages()

    result = AgentContextAssembly.describe(
        context,
        messages,
    )

    semantic_source = next(source for source in result.sources if source.item_id == "semantic-1")
    episodic_source = next(source for source in result.sources if source.item_id == "episodic-1")

    assert semantic_source.source_type == "semantic_memory"
    assert semantic_source.score == 0.91
    assert semantic_source.reranker_score == 0.87
    assert semantic_source.provenance_metadata == {
        "retrieval_method": "vector",
        "rank": 1,
    }

    assert episodic_source.source_type == "episodic_memory"
    assert episodic_source.score == 0.82
    assert episodic_source.reranker_score is None
    assert episodic_source.provenance_metadata == {
        "retrieval_method": "hybrid",
        "rank": 2,
    }


def test_describe_context_does_not_duplicate_memory_items_with_retrieval_results():
    from datetime import datetime, timezone

    from memory.models import MemoryItem
    from memory.context.builder import MemoryRetrievalResult

    _, tools, llm = _components()

    item = MemoryItem(
        id="semantic-1",
        memory_type="semantic",
        content="Known fact.",
        namespace="project-a",
        created_at=datetime.now(timezone.utc),
    )

    memory = MemoryContext(
        working=(),
        semantic=(item,),
        episodic=(),
        semantic_results=(
            MemoryRetrievalResult(
                item=item,
                retrieval_method="vector",
                rank=1,
                retrieval_score=0.95,
            ),
        ),
    )

    context = AgentContextAssembly.assemble(
        AgentRequest(
            input="Recall.",
            memory_namespace="project-a",
        ),
        tools=tools,
        llm=llm,
        memory=memory,
    )

    result = AgentContextAssembly.describe(
        context,
        context.build_llm_messages(),
    )

    semantic_sources = [
        source for source in result.sources if source.source_type == "semantic_memory"
    ]

    assert len(semantic_sources) == 1
    assert semantic_sources[0].item_id == "semantic-1"


def test_describe_context_preserves_working_memory_identity_without_scores():
    _, tools, llm = _components()

    working_item = MemoryItem(
        id="working-1",
        memory_type="working",
        content="Current task context.",
        namespace="session-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    memory = MemoryContext(
        working=(working_item,),
        semantic=(),
        episodic=(),
    )

    context = AgentContextAssembly.assemble(
        AgentRequest(
            input="Continue the task.",
            memory_namespace="session-1",
        ),
        tools=tools,
        llm=llm,
        memory=memory,
    )

    result = AgentContextAssembly.describe(
        context,
        context.build_llm_messages(),
    )

    working_sources = [
        source for source in result.sources if source.source_type == "working_memory"
    ]

    assert working_sources == [
        ContextSource(
            source_type="working_memory",
            item_id="working-1",
        )
    ]
