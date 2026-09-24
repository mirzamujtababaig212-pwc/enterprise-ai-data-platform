from unittest.mock import AsyncMock, MagicMock

import pytest

from rag.models import DocumentChunk, RetrievalResult
from rag.query import RAGQueryService


@pytest.mark.asyncio
async def test_rag_query_returns_answer_and_sources():

    retrieved = [
        RetrievalResult(
            chunk=DocumentChunk(
                id="doc-1:chunk:0",
                document_id="doc-1",
                content="The enterprise platform supports RAG.",
                metadata={
                    "source": "architecture.md",
                },
                chunk_index=0,
            ),
            score=0.95,
        ),
        RetrievalResult(
            chunk=DocumentChunk(
                id="doc-2:chunk:0",
                document_id="doc-2",
                content="The platform provides model routing.",
                metadata={
                    "source": "gateway.md",
                },
                chunk_index=0,
            ),
            score=0.81,
        ),
    ]

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(
        return_value=retrieved,
    )

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(
        return_value={
            "reply": "The platform supports RAG and model routing.",
        }
    )

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
    )

    result = await service.query(
        "What does the platform support?",
        top_k=2,
    )

    assert result.answer == ("The platform supports RAG and model routing.")

    assert result.retrieved_count == 2

    assert len(result.sources) == 2

    assert result.sources[0].document_id == "doc-1"
    assert result.sources[0].chunk_id == "doc-1:chunk:0"
    assert result.sources[0].score == 0.95

    retriever.retrieve.assert_awaited_once_with(
        "What does the platform support?",
        top_k=2,
        min_score=None,
        metadata_filter=None,
    )

    chat_service.generate.assert_awaited_once()

    generated_prompt = chat_service.generate.call_args.args[0]

    assert "What does the platform support?" in generated_prompt
    assert "The enterprise platform supports RAG." in generated_prompt
    assert "The platform provides model routing." in generated_prompt


@pytest.mark.asyncio
async def test_rag_query_passes_min_score_and_metadata_filter() -> None:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(
        return_value={
            "reply": "No relevant information was retrieved.",
        }
    )

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
    )

    metadata_filter = {
        "tenant_id": "tenant-a",
        "source": "architecture.md",
    }

    result = await service.query(
        "What does the platform support?",
        top_k=5,
        min_score=0.75,
        metadata_filter=metadata_filter,
    )

    assert result.retrieved_count == 0

    retriever.retrieve.assert_awaited_once_with(
        "What does the platform support?",
        top_k=5,
        min_score=0.75,
        metadata_filter=metadata_filter,
    )


@pytest.mark.asyncio
async def test_rag_query_scopes_retrieval_to_authenticated_tenant() -> None:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(
        return_value={"reply": "No relevant information was retrieved."}
    )

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
    )

    await service.query(
        "enterprise architecture",
        metadata_filter={"source": "architecture.md"},
        tenant_id="tenant-a",
    )

    retriever.retrieve.assert_awaited_once()

    kwargs = retriever.retrieve.call_args.kwargs

    assert kwargs["metadata_filter"] == {
        "source": "architecture.md",
    }

    policy = kwargs["governance_policy"]

    assert policy.tenant_id == "tenant-a"
    assert policy.to_metadata_filter() == {
        "tenant_id": "tenant-a",
    }


@pytest.mark.asyncio
async def test_rag_query_rejects_metadata_filter_for_different_tenant() -> None:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(return_value={"reply": "should not be generated"})

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
    )

    with pytest.raises(
        ValueError,
        match="conflicts with authenticated tenant",
    ):
        await service.query(
            "enterprise architecture",
            metadata_filter={"tenant_id": "tenant-b"},
            tenant_id="tenant-a",
        )

    retriever.retrieve.assert_not_awaited()
    chat_service.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_rag_query_honors_explicit_cross_tenant_policy() -> None:
    from ai_platform.agents.policy import TenantPolicy, TenantPolicyEngine
    from rag.governance import GovernancePolicy

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            allow_cross_tenant_data=True,
        )
    )

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(return_value={"reply": "cross tenant"})

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
        tenant_policy_engine=tenant_policy_engine,
    )

    requested_policy = GovernancePolicy(
        required_metadata={"tenant_id": "tenant-b"},
    )

    await service.query(
        "enterprise architecture",
        tenant_id="tenant-a",
        governance_policy=requested_policy,
    )

    retriever.retrieve.assert_awaited_once_with(
        "enterprise architecture",
        top_k=5,
        min_score=None,
        metadata_filter=None,
        governance_policy=requested_policy,
    )


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
async def test_rag_query_emits_governance_decision_after_retrieval() -> None:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(
        return_value={"reply": "No relevant information was retrieved."}
    )

    observer = RecordingObserver()

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
        observer=observer,
    )

    await service.query(
        "secret internal architecture query",
        top_k=5,
    )

    assert len(observer.events) == 1

    event = observer.events[0]

    from ai_platform.agents.observability import AgentExecutionEventType

    assert event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert event.agent_name == "rag.query"
    assert event.metadata == {
        "governance_domain": "rag",
        "decision": "allow",
        "retrieved_count": 0,
    }

    assert "secret internal architecture query" not in str(event.metadata)


@pytest.mark.asyncio
async def test_rag_query_governance_event_contains_tenant_policy_identity() -> None:
    from ai_platform.agents.policy import TenantPolicy, TenantPolicyEngine

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(return_value={"reply": "safe response"})

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-a",
            policy_id="rag-policy",
            policy_version="v7",
        )
    )

    observer = RecordingObserver()

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
        tenant_policy_engine=tenant_policy_engine,
        observer=observer,
    )

    await service.query(
        "tenant private query",
        tenant_id="tenant-a",
    )

    assert len(observer.events) == 1

    metadata = observer.events[0].metadata

    assert metadata == {
        "governance_domain": "rag",
        "decision": "allow",
        "retrieved_count": 0,
        "tenant_id": "tenant-a",
        "policy_id": "rag-policy",
        "policy_version": "v7",
    }

    assert "tenant private query" not in str(metadata)


@pytest.mark.asyncio
async def test_rag_query_governance_event_does_not_leak_retrieved_content() -> None:
    retrieved = [
        RetrievalResult(
            chunk=DocumentChunk(
                id="chunk-secret",
                document_id="document-secret",
                content="TOP SECRET CUSTOMER DATA",
                metadata={"source": "private.md"},
                chunk_index=0,
            ),
            score=0.99,
        )
    ]

    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=retrieved)

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(return_value={"reply": "safe answer"})

    observer = RecordingObserver()

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
        observer=observer,
    )

    await service.query("private query")

    metadata = observer.events[0].metadata

    assert metadata == {
        "governance_domain": "rag",
        "decision": "allow",
        "retrieved_count": 1,
    }

    serialized = str(metadata)

    assert "TOP SECRET CUSTOMER DATA" not in serialized
    assert "chunk-secret" not in serialized
    assert "document-secret" not in serialized
    assert "private.md" not in serialized


@pytest.mark.asyncio
async def test_rag_query_observer_failure_is_non_fatal() -> None:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(return_value=[])

    chat_service = MagicMock()
    chat_service.generate = AsyncMock(return_value={"reply": "safe response"})

    observer = FailingObserver()

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
        observer=observer,
    )

    result = await service.query("normal query")

    assert result.answer == "safe response"
    assert result.retrieved_count == 0
    assert observer.calls == 1
    retriever.retrieve.assert_awaited_once()
    chat_service.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_rag_query_does_not_emit_governance_event_when_retrieval_fails() -> None:
    retriever = MagicMock()
    retriever.retrieve = AsyncMock(side_effect=RuntimeError("retrieval unavailable"))

    chat_service = MagicMock()
    chat_service.generate = AsyncMock()

    observer = RecordingObserver()

    service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
        observer=observer,
    )

    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        await service.query("query")

    assert observer.events == []
    chat_service.generate.assert_not_awaited()
