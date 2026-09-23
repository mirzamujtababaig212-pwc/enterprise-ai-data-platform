from __future__ import annotations

import pytest

from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.policy import TenantPolicy, TenantPolicyEngine
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.llm_gateway.routing.router import Router
from rag.chunking.recursive import RecursiveChunker
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.indexing import RAGIndexer
from rag.models import Document
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore
from tools.execution.service import ToolExecutionService
from tools.rag.search import RAGSearchTool
from tools.registry.in_memory import InMemoryToolRegistry


@pytest.mark.asyncio
async def test_agent_runtime_executes_rag_search_tool() -> None:
    gateway = Router()

    vector_store = InMemoryVectorStore()

    embedding_service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
        gateway_router=gateway,
    )

    indexer = RAGIndexer(
        chunker=RecursiveChunker(
            chunk_size=500,
            overlap=50,
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    await indexer.index(
        Document(
            id="agent-rag-doc",
            content=(
                "The Enterprise AI Platform uses a unified LLM Gateway "
                "for model routing, usage tracking, observability, and "
                "enterprise AI workloads."
            ),
            metadata={
                "source": "architecture.md",
            },
        )
    )

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    tool_registry = InMemoryToolRegistry()
    await tool_registry.register(RAGSearchTool(retriever))

    agent_definition = AgentDefinition(
        name="enterprise-rag-analyst-test",
        description="RAG-enabled integration test agent.",
        system_prompt=(
            "You are an enterprise RAG analyst. "
            "Use the rag.search tool when enterprise knowledge is required."
        ),
        model="mock-gpt",
        temperature=0.2,
        max_tokens=1024,
        tool_names=("rag.search",),
    )

    agent_registry = InMemoryAgentRegistry()

    await agent_registry.register(LLMAgent(agent_definition))

    runtime = AgentRuntime(
        agent_registry,
        tool_registry=tool_registry,
        llm_gateway=gateway,
    )

    response = await runtime.run(
        "enterprise-rag-analyst-test",
        AgentRequest(
            input="What does the Enterprise AI Platform use for model routing?",
            metadata={
                "mock_tool_call": "rag.search",
            },
        ),
    )

    assert response.agent_name == "enterprise-rag-analyst-test"
    assert response.metadata["provider"] == "mock"
    assert response.metadata["model"] == "mock-gpt"
    assert response.metadata["tool_rounds"] == 1
    assert "The Enterprise AI Platform uses a unified LLM Gateway" in response.output


class TenantPolicyToolCallingLLMGateway:
    """Deterministic LLM gateway that requests rag.search once."""

    def __init__(self) -> None:
        self.requests: list[dict] = []

    async def route_chat(self, request: dict) -> dict:
        self.requests.append(request)

        if len(self.requests) == 1:
            from ai_platform.agents.tool_calls import AgentToolCall

            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="tenant-policy-rag-call-1",
                        name="rag.search",
                        arguments={"query": "RAG"},
                    )
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "RAG retrieves relevant enterprise context.",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
            },
        }


class CountingRAGTool(RAGSearchTool):
    def __init__(self) -> None:
        self.execution_count = 0

        class _Retriever:
            async def search(self, query, **kwargs):
                return []

        super().__init__(_Retriever())

    async def execute(self, arguments):
        self.execution_count += 1
        return await self._result(arguments)

    async def execute_with_context(self, arguments, context):
        self.execution_count += 1
        return await self._result(arguments)

    async def _result(self, arguments):
        return {
            "query": arguments["query"],
            "retrieved_count": 1,
            "results": [
                {
                    "chunk_id": "tenant-policy-rag-001",
                    "document_id": "tenant-policy-doc-001",
                    "content": "Tenant-authorized RAG context.",
                    "score": 0.95,
                    "metadata": {
                        "tenant_id": "tenant-acme",
                    },
                }
            ],
        }


@pytest.mark.asyncio
async def test_agent_llm_tool_call_enforces_tenant_policy_and_persists_step() -> None:
    from app.control_plane.agent_run_steps.in_memory import (
        InMemoryAgentRunStepsRepository,
    )

    gateway = TenantPolicyToolCallingLLMGateway()
    tool_registry = InMemoryToolRegistry()
    tool = CountingRAGTool()

    await tool_registry.register(tool)

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_tools=frozenset({"rag.search"}),
        )
    )

    execution_service = ToolExecutionService(
        tool_registry,
        tenant_policy_engine=tenant_policy_engine,
    )

    agent_definition = AgentDefinition(
        name="enterprise-rag-analyst",
        description="Tenant-policy RAG integration test agent.",
        system_prompt=("Use rag.search when enterprise knowledge is required."),
        model="mock-gpt",
        tool_names=("rag.search",),
    )

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(LLMAgent(agent_definition))

    repository = InMemoryAgentRunStepsRepository()

    runtime = AgentRuntime(
        agent_registry,
        tool_registry=tool_registry,
        tool_execution_service=execution_service,
        llm_gateway=gateway,
        agent_run_steps_repository_factory=lambda: repository,
    )

    run_id = "tenant-policy-rag-allowed"

    response = await runtime.run(
        "enterprise-rag-analyst",
        AgentRequest(
            input="Find RAG information.",
            session_id="tenant-policy-session",
            tenant_id="tenant-acme",
        ),
        run_id=run_id,
    )

    assert response.agent_name == "enterprise-rag-analyst"
    assert response.output == "RAG retrieves relevant enterprise context."
    assert response.metadata["provider"] == "fake"
    assert response.metadata["tool_rounds"] == 0

    # The tenant identity must travel from AgentRequest all the way to
    # ToolExecutionService/TenantPolicyEngine.
    assert len(gateway.requests) == 3

    # The underlying tool must execute exactly once after policy approval.
    assert tool.execution_count == 1

    step = repository.get(run_id, "retrieve_evidence")

    assert step is not None
    assert step.status.value == "completed"
    assert step.tool_name == "rag.search"
    assert step.call_id == "tenant-policy-rag-call-1"
    assert step.input == {"query": "RAG"}
    assert step.output["retrieved_count"] == 1


@pytest.mark.asyncio
async def test_agent_llm_tool_call_is_denied_by_tenant_policy_before_tool_execution() -> None:
    gateway = TenantPolicyToolCallingLLMGateway()
    tool_registry = InMemoryToolRegistry()
    tool = CountingRAGTool()

    await tool_registry.register(tool)

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_tools=frozenset({"rag.search"}),
        )
    )

    execution_service = ToolExecutionService(
        tool_registry,
        tenant_policy_engine=tenant_policy_engine,
    )

    agent_definition = AgentDefinition(
        name="tenant-policy-rag-denied-agent",
        description="Tenant-policy RAG denial integration test agent.",
        system_prompt=("Use rag.search when enterprise knowledge is required."),
        model="mock-gpt",
        tool_names=("rag.search",),
    )

    agent_registry = InMemoryAgentRegistry()
    await agent_registry.register(LLMAgent(agent_definition))

    runtime = AgentRuntime(
        agent_registry,
        tool_registry=tool_registry,
        tool_execution_service=execution_service,
        llm_gateway=gateway,
    )

    response = await runtime.run(
        "tenant-policy-rag-denied-agent",
        AgentRequest(
            input="Find RAG information.",
            session_id="tenant-policy-denied-session",
            tenant_id="tenant-other",
        ),
        run_id="tenant-policy-rag-denied",
    )

    assert response.agent_name == "tenant-policy-rag-denied-agent"

    # The policy engine rejects the tool request because tenant-other has
    # no registered policy.
    assert tool.execution_count == 0

    # The LLM receives the policy failure as its tool result and produces
    # its second response without the underlying tool ever executing.
    assert len(gateway.requests) == 2
    assert response.metadata["tool_rounds"] == 1
