from __future__ import annotations

import asyncio

from fastapi import Depends
from sqlalchemy.orm import Session

from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.models import AgentDefinition
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.llm_gateway.config.settings import settings
from ai_platform.llm_gateway.routing.router import Router
from ml.inference import VehicleRiskPredictor
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.generation.gateway import GatewayChatService
from rag.indexing import RAGIndexer
from rag.chunking.recursive import RecursiveChunker
from rag.query import RAGQueryService
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore
from tools.registry.in_memory import InMemoryToolRegistry
from tools.rag.search import RAGSearchTool

from app.control_plane.persistence.database import get_db
from app.control_plane.persistence.rag_state import PostgreSQLRAGStateRepository
from app.control_plane.usage.postgres_store import PostgreSQLUsageRepository

_llm_router = Router()

_agent_registry = InMemoryAgentRegistry()
_tool_registry = InMemoryToolRegistry()

_agent_runtime = AgentRuntime(
    _agent_registry,
    tool_registry=_tool_registry,
    llm_gateway=_llm_router,
)

_agent_initialization_lock = asyncio.Lock()
_agents_initialized = False

_vehicle_risk_predictor = VehicleRiskPredictor(
    model_alias="champion",
)

_rag_vector_store = InMemoryVectorStore()

_rag_chunker = RecursiveChunker(
    chunk_size=1000,
    overlap=100,
)

_rag_embedding_service = GatewayEmbeddingService(
    provider=settings.DEFAULT_PROVIDER,
    model=settings.DEFAULT_EMBEDDING_MODEL,
    gateway_router=_llm_router,
)

_rag_chat_service = GatewayChatService(
    provider=settings.DEFAULT_PROVIDER,
    model=settings.DEFAULT_CHAT_MODEL,
    gateway_router=_llm_router,
)

_rag_indexer = RAGIndexer(
    chunker=_rag_chunker,
    embedding_service=_rag_embedding_service,
    vector_store=_rag_vector_store,
)

_rag_retriever = SemanticRetriever(
    embedding_service=_rag_embedding_service,
    vector_store=_rag_vector_store,
)

_rag_query_service = RAGQueryService(
    retriever=_rag_retriever,
    chat_service=_rag_chat_service,
)


async def _initialize_agents() -> None:
    global _agents_initialized

    if _agents_initialized:
        return

    async with _agent_initialization_lock:
        if _agents_initialized:
            return

        rag_search_tool = RAGSearchTool(_rag_retriever)
        await _tool_registry.register(rag_search_tool)

        analyst_definition = AgentDefinition(
            name="enterprise-analyst",
            description=(
                "Enterprise AI analyst for reasoning, explanation, "
                "analysis, and platform-oriented tasks."
            ),
            system_prompt=(
                "You are an enterprise AI analyst. "
                "Provide accurate, structured, concise, and technically "
                "grounded responses. Use only the capabilities explicitly "
                "available to you."
            ),
            model=settings.DEFAULT_CHAT_MODEL,
            temperature=0.2,
            max_tokens=2048,
        )

        rag_analyst_definition = AgentDefinition(
            name="enterprise-rag-analyst",
            description=(
                "Enterprise AI analyst with access to the enterprise "
                "knowledge base through semantic retrieval."
            ),
            system_prompt=(
                "You are an enterprise RAG analyst. "
                "Use the rag.search tool when relevant enterprise "
                "knowledge is needed. Ground your answer in retrieved "
                "sources and clearly distinguish retrieved information "
                "from general reasoning. Do not invent facts that are "
                "not supported by the available context."
            ),
            model=settings.DEFAULT_CHAT_MODEL,
            temperature=0.2,
            max_tokens=2048,
            tool_names=("rag.search",),
        )

        await _agent_registry.register(LLMAgent(analyst_definition))
        await _agent_registry.register(LLMAgent(rag_analyst_definition))

        _agents_initialized = True


async def get_agent_runtime() -> AgentRuntime:
    await _initialize_agents()
    return _agent_runtime


def get_llm_router() -> Router:
    return _llm_router


def get_vehicle_risk_predictor() -> VehicleRiskPredictor:
    return _vehicle_risk_predictor


def get_rag_vector_store() -> InMemoryVectorStore:
    return _rag_vector_store


def get_rag_query_service() -> RAGQueryService:
    return _rag_query_service


def get_rag_indexer() -> RAGIndexer:
    return _rag_indexer


def get_rag_state_repository(
    db: Session = Depends(get_db),
) -> PostgreSQLRAGStateRepository:
    return PostgreSQLRAGStateRepository(db)


def get_usage_store(
    db: Session = Depends(get_db),
) -> PostgreSQLUsageRepository:
    return PostgreSQLUsageRepository(db)
