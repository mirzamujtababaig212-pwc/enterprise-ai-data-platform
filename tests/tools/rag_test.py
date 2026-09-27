from __future__ import annotations

import pytest

from rag.models import DocumentChunk, RetrievalResult
from tools.rag.search import RAGSearchTool
from tools.execution.context import ToolExecutionContext
from rag.governance import GovernancePolicy


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                str,
                int,
                float | None,
                dict[str, object] | None,
                GovernancePolicy | None,
            ]
        ] = []

    async def retrieve(
        self,
        query: str,
        top_k: int,
        min_score: float | None = None,
        metadata_filter: dict[str, object] | None = None,
        governance_policy: GovernancePolicy | None = None,
    ) -> list[RetrievalResult]:
        self.calls.append(
            (
                query,
                top_k,
                min_score,
                metadata_filter,
                governance_policy,
            )
        )

        return [
            RetrievalResult(
                chunk=DocumentChunk(
                    id="chunk-1",
                    document_id="doc-1",
                    content="Enterprise RAG retrieves relevant knowledge.",
                    metadata={"source": "test"},
                    chunk_index=0,
                ),
                score=0.95,
            ),
            RetrievalResult(
                chunk=DocumentChunk(
                    id="chunk-2",
                    document_id="doc-1",
                    content="Retrieved context is supplied to the agent.",
                    metadata={"source": "test"},
                    chunk_index=1,
                ),
                score=0.85,
            ),
        ]


@pytest.mark.asyncio
async def test_rag_search_returns_structured_results() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    result = await tool.execute(
        {
            "query": "How does enterprise RAG work?",
            "top_k": 2,
        }
    )

    assert retriever.calls == [
        (
            "How does enterprise RAG work?",
            2,
            None,
            None,
            None,
        ),
    ]

    assert result["query"] == "How does enterprise RAG work?"
    assert result["retrieved_count"] == 2
    assert result["results"] == [
        {
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "content": "Enterprise RAG retrieves relevant knowledge.",
            "score": 0.95,
            "retrieval_score": 0.95,
            "reranker_score": None,
            "metadata": {"source": "test"},
        },
        {
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "content": "Retrieved context is supplied to the agent.",
            "score": 0.85,
            "retrieval_score": 0.85,
            "reranker_score": None,
            "metadata": {"source": "test"},
        },
    ]


@pytest.mark.asyncio
async def test_rag_search_defaults_top_k_to_five() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    await tool.execute({"query": "RAG"})

    assert retriever.calls == [
        (
            "RAG",
            5,
            None,
            None,
            None,
        )
    ]


@pytest.mark.asyncio
async def test_rag_search_rejects_empty_query() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="query must be a non-empty string",
    ):
        await tool.execute({"query": "   "})


@pytest.mark.asyncio
async def test_rag_search_rejects_invalid_top_k() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    with pytest.raises(
        ValueError,
        match="top_k must be between 1 and 10",
    ):
        await tool.execute(
            {
                "query": "RAG",
                "top_k": 11,
            }
        )


def test_rag_search_definition() -> None:
    retriever = FakeRetriever()
    tool = RAGSearchTool(retriever)  # type: ignore[arg-type]

    definition = tool.definition

    assert definition.name == "rag.search"
    assert "knowledge base" in definition.description
    assert definition.input_schema["required"] == ["query"]
    assert definition.input_schema["properties"]["top_k"]["default"] == 5
    assert definition.metadata["category"] == "retrieval"


@pytest.mark.asyncio
async def test_rag_search_passes_min_score_and_metadata_filter() -> None:
    retriever = FakeRetriever()

    tool = RAGSearchTool(retriever)

    metadata_filter = {
        "tenant_id": "tenant-a",
        "source": "architecture.md",
    }

    result = await tool.execute(
        {
            "query": "enterprise architecture",
            "top_k": 5,
            "min_score": 0.75,
            "metadata_filter": metadata_filter,
        }
    )

    assert result["retrieved_count"] == 2

    assert retriever.calls == [
        (
            "enterprise architecture",
            5,
            0.75,
            metadata_filter,
            None,
        )
    ]


@pytest.mark.asyncio
async def test_rag_search_rejects_invalid_min_score():
    tool = RAGSearchTool(FakeRetriever())

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        await tool.execute(
            {
                "query": "RAG",
                "min_score": 1.1,
            }
        )


@pytest.mark.asyncio
async def test_rag_search_rejects_invalid_metadata_filter():
    tool = RAGSearchTool(FakeRetriever())

    with pytest.raises(
        TypeError,
        match="metadata_filter must be an object",
    ):
        await tool.execute(
            {
                "query": "RAG",
                "metadata_filter": ["tenant-a"],
            }
        )


@pytest.mark.asyncio
async def test_rag_search_passes_governance_policy_from_context() -> None:
    retriever = FakeRetriever()

    tool = RAGSearchTool(retriever)

    policy = GovernancePolicy(
        required_metadata={
            "tenant_id": "tenant-a",
        },
    )

    await tool.execute_with_context(
        {
            "query": "enterprise architecture",
        },
        ToolExecutionContext(
            governance_policy=policy,
        ),
    )

    assert retriever.calls[0][4] is policy


class RecordingObserver:
    def __init__(self) -> None:
        self.events = []

    async def record(self, event) -> None:
        self.events.append(event)


class FailingObserver:
    def __init__(self) -> None:
        self.calls = 0

    async def record(self, event) -> None:
        self.calls += 1
        raise RuntimeError("event persistence unavailable")


@pytest.mark.asyncio
async def test_rag_search_emits_single_governance_event_with_tool_context() -> None:
    from ai_platform.agents.observability import AgentExecutionEventType
    from ai_platform.agents.policy import TenantPolicy, TenantPolicyEngine

    retriever = FakeRetriever()
    observer = RecordingObserver()

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            policy_id="rag-policy",
            policy_version="v3",
        )
    )

    tool = RAGSearchTool(
        retriever,
        observer=observer,
        tenant_policy_engine=tenant_policy_engine,
    )

    context = ToolExecutionContext(
        run_id="run-123",
        call_id="call-456",
        agent_name="enterprise-analyst",
        session_id="session-789",
        user_id="user-001",
        tenant_id="tenant-a",
        governance_policy=GovernancePolicy(
            required_metadata={"tenant_id": "tenant-a"},
        ),
    )

    result = await tool.execute_with_context(
        {
            "query": "TOP SECRET CUSTOMER QUERY",
            "top_k": 2,
        },
        context,
    )

    assert result["retrieved_count"] == 2
    assert len(observer.events) == 1

    event = observer.events[0]

    assert event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert event.agent_name == "rag.search"
    assert event.run_id == "run-123"
    assert event.session_id == "session-789"
    assert event.user_id == "user-001"

    assert event.metadata == {
        "governance_domain": "rag",
        "decision": "allow",
        "retrieved_count": 2,
        "tenant_id": "tenant-a",
        "policy_id": "rag-policy",
        "policy_version": "v3",
    }

    serialized = str(event.metadata)

    assert "TOP SECRET CUSTOMER QUERY" not in serialized
    assert "Enterprise RAG retrieves relevant knowledge." not in serialized
    assert "chunk-1" not in serialized
    assert "doc-1" not in serialized


@pytest.mark.asyncio
async def test_rag_search_direct_execution_emits_at_most_one_governance_event() -> None:
    retriever = FakeRetriever()
    observer = RecordingObserver()

    tool = RAGSearchTool(
        retriever,
        observer=observer,
    )

    result = await tool.execute(
        {
            "query": "RAG",
            "top_k": 2,
        }
    )

    assert result["retrieved_count"] == 2
    assert len(observer.events) == 1

    event = observer.events[0]

    assert event.agent_name == "rag.search"
    assert event.run_id is None
    assert event.session_id is None
    assert event.user_id is None
    assert event.metadata == {
        "governance_domain": "rag",
        "decision": "allow",
        "retrieved_count": 2,
    }


@pytest.mark.asyncio
async def test_rag_search_governance_observer_failure_is_non_fatal() -> None:
    retriever = FakeRetriever()
    observer = FailingObserver()

    tool = RAGSearchTool(
        retriever,
        observer=observer,
    )

    result = await tool.execute(
        {
            "query": "RAG",
            "top_k": 2,
        }
    )

    assert result["retrieved_count"] == 2
    assert observer.calls == 1
    assert len(retriever.calls) == 1


@pytest.mark.asyncio
async def test_rag_search_does_not_emit_governance_event_when_retrieval_fails() -> None:
    class FailingRetriever:
        async def retrieve(
            self,
            query,
            top_k,
            min_score=None,
            metadata_filter=None,
            governance_policy=None,
        ):
            raise RuntimeError("retrieval unavailable")

    observer = RecordingObserver()
    tool = RAGSearchTool(
        FailingRetriever(),
        observer=observer,
    )

    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        await tool.execute({"query": "RAG"})

    assert observer.events == []


@pytest.mark.asyncio
async def test_rag_search_governance_event_does_not_duplicate_for_retriever_calls() -> None:
    retriever = FakeRetriever()
    observer = RecordingObserver()

    tool = RAGSearchTool(
        retriever,
        observer=observer,
    )

    await tool.execute_with_context(
        {"query": "RAG", "top_k": 2},
        ToolExecutionContext(
            run_id="run-1",
            call_id="call-1",
            tenant_id="tenant-a",
        ),
    )

    assert len(retriever.calls) == 1
    assert len(observer.events) == 1
