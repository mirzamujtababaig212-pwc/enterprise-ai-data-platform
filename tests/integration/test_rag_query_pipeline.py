import pytest

from ai_platform.llm_gateway.routing.router import Router
from rag.chunking import RecursiveChunker
from rag.embeddings import GatewayEmbeddingService
from rag.generation import GatewayChatService
from rag.indexing import RAGIndexer
from rag.models import Document
from rag.query import RAGQueryService
from rag.retrieval import SemanticRetriever
from rag.stores import InMemoryVectorStore


@pytest.mark.asyncio
async def test_rag_end_to_end_with_mock_gateway():

    vector_store = InMemoryVectorStore()

    embedding_service = GatewayEmbeddingService(
        provider="mock",
        model="mock-embedding",
    )

    chunker = RecursiveChunker(
        chunk_size=500,
        overlap=50,
    )

    indexer = RAGIndexer(
        chunker=chunker,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    document = Document(
        id="enterprise-ai-os",
        content=(
            "The Enterprise AI OS provides retrieval augmented "
            "generation, model routing, enterprise observability, "
            "agent execution, and governance."
        ),
        metadata={
            "source": "enterprise-ai-os.md",
            "type": "architecture",
        },
    )

    indexed = await indexer.index(
        document,
    )

    assert indexed

    retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    router = Router()

    chat_service = GatewayChatService(
        provider="mock",
        model="mock-gpt",
        gateway_router=router,
    )

    rag_service = RAGQueryService(
        retriever=retriever,
        chat_service=chat_service,
    )

    result = await rag_service.query(
        "What does the Enterprise AI OS provide?",
        top_k=3,
    )

    assert result.answer

    assert result.retrieved_count > 0

    assert result.sources

    assert result.sources[0].document_id == "enterprise-ai-os"
