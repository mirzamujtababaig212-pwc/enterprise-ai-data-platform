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

_llm_router = Router()

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
