from __future__ import annotations

import asyncio

from fastapi import Depends
from sqlalchemy.orm import Session

from ai_platform.agents.composite_observer import CompositeAgentExecutionObserver
from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.models import AgentDefinition
from ai_platform.agents.otel_observer import OpenTelemetryAgentExecutionObserver
from ai_platform.agents.prometheus_observer import PrometheusAgentExecutionObserver
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.llm_gateway.config.settings import settings
from ai_platform.llm_gateway.routing.router import Router
from ml.inference import VehicleRiskPredictor
from memory import MemoryService
from memory.embeddings.postgres import PostgreSQLMemoryEmbeddingStore
from memory.context import MemoryContextBuilder
from memory.retrieval.factory import MemoryRetrieverFactory
from memory.stores.factory import MemoryStoreFactory
from rag.embeddings.gateway import GatewayEmbeddingService
from rag.generation.gateway import GatewayChatService
from rag.indexing import RAGIndexer
from rag.chunking.recursive import RecursiveChunker
from rag.query import RAGQueryService
from rag.retrieval.factory import RAGRetrieverFactory
from rag.retrieval.lexical import PostgreSQLLexicalRetriever
from rag.retrieval.retriever import SemanticRetriever
from rag.contracts import Retriever, VectorStore
from tools.registry.in_memory import InMemoryToolRegistry
from tools.rag.search import RAGSearchTool

from app.config.settings import Settings
from common.config.settings import Settings as CommonSettings
from rag.evaluation.external.dispatcher import (
    ExternalEvaluationDispatcher,
    RagasExternalEvaluationDispatcher,
)
from rag.evaluation.external.release import ExternalEvaluationReleasePolicy
from rag.evaluation.external.workflow import RAGGenerationEvaluationWorkflow
from app.control_plane.persistence.database import SessionLocal, get_db
from app.control_plane.agent_run_events.postgres_observer import (
    PostgreSQLAgentRunEventObserver,
)
from app.control_plane.agent_run_events.postgres_repository import (
    PostgreSQLAgentRunEventsRepository,
)
from app.control_plane.agent_checkpoints.postgres_handler import (
    PostgreSQLAgentCheckpointHandler,
)
from app.control_plane.agent_run_events.tool_authorization_observer import (
    ToolAuthorizationAuditObserver,
)
from app.control_plane.tool_execution.postgres_idempotency import (
    PostgreSQLToolExecutionIdempotencyStore,
)
from tools.execution.service import ToolExecutionService
from app.control_plane.agent_runs.application_service import AgentRunApplicationService
from app.control_plane.agent_runs.cancellation import AgentRunCancellationRegistry
from app.control_plane.agent_runs.postgres_repository import PostgreSQLAgentRunRepository
from app.control_plane.evaluation_application_service import EvaluationApplicationService
from app.control_plane.evaluation_service import EvaluationExecutionService
from app.control_plane.persistence.rag_state import PostgreSQLRAGStateRepository
from rag.evaluation.stores.postgres import PostgreSQLRetrievalEvaluationRunStore
from rag.evaluation.stores.release_decision import (
    PostgreSQLRetrievalEvaluationReleaseDecisionStore,
)
from app.control_plane.usage.postgres_store import PostgreSQLUsageRepository
from rag.stores.factory import VectorStoreFactory
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)
from app.control_plane.agent_runs.recovery_service import (
    AgentRunRecoveryService,
)

_llm_router = Router()

_agent_registry = InMemoryAgentRegistry()
_tool_registry = InMemoryToolRegistry()
_agent_run_cancellation_registry = AgentRunCancellationRegistry()

_agent_observer = CompositeAgentExecutionObserver(
    [
        OpenTelemetryAgentExecutionObserver(),
        PrometheusAgentExecutionObserver(),
        PostgreSQLAgentRunEventObserver(SessionLocal),
    ]
)
_agent_checkpoint_handler = PostgreSQLAgentCheckpointHandler(SessionLocal)

_tool_authorization_audit_sink = ToolAuthorizationAuditObserver(
    _agent_observer,
)

_tool_idempotency_store = PostgreSQLToolExecutionIdempotencyStore(
    SessionLocal,
)

_tool_execution_service = ToolExecutionService(
    _tool_registry,
    audit_sink=_tool_authorization_audit_sink,
    idempotency_store=_tool_idempotency_store,
)

_memory_store = MemoryStoreFactory.create()

_rag_embedding_service = GatewayEmbeddingService(
    provider=settings.DEFAULT_PROVIDER,
    model=settings.DEFAULT_EMBEDDING_MODEL,
    gateway_router=_llm_router,
)

_memory_embedding_store = (
    PostgreSQLMemoryEmbeddingStore() if CommonSettings.memory_store.BACKEND == "postgres" else None
)

_memory_service = MemoryService(
    _memory_store,
    embedding_service=_rag_embedding_service if _memory_embedding_store is not None else None,
    embedding_store=_memory_embedding_store,
)
_memory_retriever = MemoryRetrieverFactory.create(
    backend=CommonSettings.memory_store.BACKEND,
    memory_store=_memory_store,
    embedding_service=_rag_embedding_service,
    reranker=CommonSettings.memory_store.RERANKER,
    reranker_model_id=CommonSettings.memory_store.RERANKER_MODEL_ID,
    reranker_onnx_filename=CommonSettings.memory_store.RERANKER_ONNX_FILENAME,
    reranker_max_length=CommonSettings.memory_store.RERANKER_MAX_LENGTH,
    reranker_candidate_k=CommonSettings.memory_store.RERANKER_CANDIDATE_K,
)
_memory_context_builder = MemoryContextBuilder(
    _memory_service,
    memory_retriever=_memory_retriever,
)

_agent_runtime = AgentRuntime(
    _agent_registry,
    tool_registry=_tool_registry,
    tool_execution_service=_tool_execution_service,
    llm_gateway=_llm_router,
    memory_context_builder=_memory_context_builder,
    memory_service=_memory_service,
)

_agent_initialization_lock = asyncio.Lock()
_agents_initialized = False

_vehicle_risk_predictor = VehicleRiskPredictor(
    model_alias="champion",
)


def _build_rag_retriever(
    *,
    semantic_retriever: SemanticRetriever,
    backend: str,
    lexical_retriever: PostgreSQLLexicalRetriever | None = None,
) -> Retriever:
    return RAGRetrieverFactory.create(
        backend=backend,
        semantic_retriever=semantic_retriever,
        lexical_retriever=lexical_retriever,
        reranker=CommonSettings.rag_retrieval.RERANKER,
        reranker_model_id=CommonSettings.rag_retrieval.RERANKER_MODEL_ID,
        reranker_onnx_filename=CommonSettings.rag_retrieval.RERANKER_ONNX_FILENAME,
        reranker_max_length=CommonSettings.rag_retrieval.RERANKER_MAX_LENGTH,
        reranker_candidate_k=CommonSettings.rag_retrieval.RERANKER_CANDIDATE_K,
    )


_rag_vector_store = VectorStoreFactory.create()

_rag_chunker = RecursiveChunker(
    chunk_size=1000,
    overlap=100,
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

_rag_semantic_retriever = SemanticRetriever(
    embedding_service=_rag_embedding_service,
    vector_store=_rag_vector_store,
)

_rag_lexical_retriever = (
    PostgreSQLLexicalRetriever()
    if CommonSettings.vector_store.BACKEND in {"postgres", "qdrant"}
    else None
)

_rag_retriever = _build_rag_retriever(
    semantic_retriever=_rag_semantic_retriever,
    backend=CommonSettings.vector_store.BACKEND,
    lexical_retriever=_rag_lexical_retriever,
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

        await _agent_registry.register(
            LLMAgent(
                analyst_definition,
                observer=_agent_observer,
                checkpoint_handler=_agent_checkpoint_handler,
            )
        )
        await _agent_registry.register(
            LLMAgent(
                rag_analyst_definition,
                observer=_agent_observer,
                checkpoint_handler=_agent_checkpoint_handler,
            )
        )

        _agents_initialized = True


async def get_agent_runtime() -> AgentRuntime:
    await _initialize_agents()
    return _agent_runtime


async def get_agent_run_application_service(
    db: Session = Depends(get_db),
) -> AgentRunApplicationService:
    await _initialize_agents()
    app_settings = Settings.from_environment()

    return AgentRunApplicationService(
        runtime=_agent_runtime,
        repository=PostgreSQLAgentRunRepository(db),
        events_repository=PostgreSQLAgentRunEventsRepository(db),
        observer=_agent_observer,
        cancellation_registry=_agent_run_cancellation_registry,
        lease_seconds=app_settings.agent_run_lease_duration_seconds,
    )


async def get_agent_run_recovery_service(
    db: Session = Depends(get_db),
) -> AgentRunRecoveryService:
    await _initialize_agents()
    app_settings = Settings.from_environment()

    return AgentRunRecoveryService(
        runtime=_agent_runtime,
        repository=PostgreSQLAgentRunRepository(db),
        checkpoints_repository=PostgreSQLAgentCheckpointsRepository(db),
        observer=_agent_observer,
        lease_seconds=app_settings.agent_run_lease_duration_seconds,
    )


async def build_agent_run_recovery_service(
    db: Session,
) -> AgentRunRecoveryService:
    await _initialize_agents()
    app_settings = Settings.from_environment()

    return AgentRunRecoveryService(
        runtime=_agent_runtime,
        repository=PostgreSQLAgentRunRepository(db),
        checkpoints_repository=PostgreSQLAgentCheckpointsRepository(db),
        observer=_agent_observer,
        lease_seconds=app_settings.agent_run_lease_duration_seconds,
    )


def get_llm_router() -> Router:
    return _llm_router


def get_vehicle_risk_predictor() -> VehicleRiskPredictor:
    return _vehicle_risk_predictor


def get_rag_vector_store() -> VectorStore:
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


def get_retrieval_evaluation_run_store(
    db: Session = Depends(get_db),
) -> PostgreSQLRetrievalEvaluationRunStore:
    return PostgreSQLRetrievalEvaluationRunStore(db)


def get_retrieval_evaluation_release_decision_store(
    db: Session = Depends(get_db),
) -> PostgreSQLRetrievalEvaluationReleaseDecisionStore:
    return PostgreSQLRetrievalEvaluationReleaseDecisionStore(db)


def get_external_evaluation_release_policy() -> ExternalEvaluationReleasePolicy:
    settings = Settings.from_environment()

    return ExternalEvaluationReleasePolicy(
        name="application-external-evaluation-release",
        required=settings.external_evaluation_release_required,
    )


async def close_rag_vector_store() -> None:
    close = getattr(_rag_vector_store, "close", None)

    if close is not None:
        await close()


def get_external_evaluation_dispatcher() -> ExternalEvaluationDispatcher:
    """
    Construct a provider-neutral external evaluation dispatcher.

    RAGAS remains an optional dependency, so its concrete classes and workflow
    are imported and constructed only when a RAGAS evaluation is explicitly
    requested.
    """

    def build_faithfulness_workflow() -> RAGGenerationEvaluationWorkflow:
        try:
            from rag.evaluation.external.ragas import (
                GatewayRagasLLM,
                RagasFaithfulnessAdapter,
            )
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS external evaluation requires the optional dependency. "
                'Install it with: pip install -e ".[ragas]"'
            ) from exc

        llm = GatewayRagasLLM(
            _rag_chat_service,
        )

        evaluator = RagasFaithfulnessAdapter(
            llm=llm,
        )

        return RAGGenerationEvaluationWorkflow(
            chat_service=_rag_chat_service,
            evaluator=evaluator,
        )

    def build_answer_relevancy_workflow() -> RAGGenerationEvaluationWorkflow:
        try:
            from rag.evaluation.external.ragas import (
                GatewayRagasEmbedding,
                GatewayRagasLLM,
                RagasAnswerRelevancyAdapter,
            )
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS external evaluation requires the optional dependency. "
                'Install it with: pip install -e ".[ragas]"'
            ) from exc

        llm = GatewayRagasLLM(
            _rag_chat_service,
        )

        embeddings = GatewayRagasEmbedding(
            _rag_embedding_service,
        )

        evaluator = RagasAnswerRelevancyAdapter(
            llm=llm,
            embeddings=embeddings,
        )

        return RAGGenerationEvaluationWorkflow(
            chat_service=_rag_chat_service,
            evaluator=evaluator,
        )

    return RagasExternalEvaluationDispatcher(
        faithfulness_workflow_factory=build_faithfulness_workflow,
        answer_relevancy_workflow_factory=build_answer_relevancy_workflow,
    )


def get_evaluation_application_service(
    db: Session = Depends(get_db),
) -> EvaluationApplicationService:
    external_evaluation_dispatcher = get_external_evaluation_dispatcher()

    return EvaluationApplicationService(
        execution_service=EvaluationExecutionService(
            session=db,
            external_evaluation_dispatcher=external_evaluation_dispatcher,
        ),
        run_store=PostgreSQLRetrievalEvaluationRunStore(db),
    )
