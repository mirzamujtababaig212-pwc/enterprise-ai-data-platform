from __future__ import annotations

import pytest

from ai_platform.agents.llm_agent import LLMAgent
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
