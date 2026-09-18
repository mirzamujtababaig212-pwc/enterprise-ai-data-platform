import pytest

from rag.governance import GovernancePolicy
from rag.models import RetrievalResult
from rag.retrieval.factory import RAGRetrieverFactory
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.reranker import RerankingRetriever


class FakeRetriever:
    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float | None = None,
        metadata_filter: dict[str, object] | None = None,
        governance_policy: GovernancePolicy | None = None,
    ) -> list[RetrievalResult]:
        return []


class FakeReranker:
    DEFAULT_MODEL_ID = "fake-model"
    DEFAULT_ONNX_FILENAME = "fake.onnx"

    def __init__(
        self,
        *,
        model_id: str,
        onnx_filename: str,
        max_length: int,
    ) -> None:
        self.model_id = model_id
        self.onnx_filename = onnx_filename
        self.max_length = max_length


def _semantic_retriever() -> FakeRetriever:
    return FakeRetriever()


def _lexical_retriever() -> FakeRetriever:
    return FakeRetriever()


def test_in_memory_returns_semantic_retriever():
    semantic = _semantic_retriever()

    result = RAGRetrieverFactory.create(
        backend="in_memory",
        semantic_retriever=semantic,
    )

    assert result is semantic


def test_in_memory_rejects_cross_encoder():
    with pytest.raises(ValueError, match="hybrid RAG backend"):
        RAGRetrieverFactory.create(
            backend="in_memory",
            semantic_retriever=_semantic_retriever(),
            reranker="cross_encoder",
        )


def test_hybrid_backend_returns_hybrid_retriever_without_reranker():
    semantic = _semantic_retriever()
    lexical = _lexical_retriever()

    result = RAGRetrieverFactory.create(
        backend="postgres",
        semantic_retriever=semantic,
        lexical_retriever=lexical,
    )

    assert isinstance(result, HybridRetriever)
    assert result.semantic_retriever is semantic
    assert result.lexical_retriever is lexical
    assert result.candidate_k == 5
    assert result.rrf_k == 60
    assert result.semantic_weight == 1.0
    assert result.lexical_weight == 0.5


def test_qdrant_backend_uses_hybrid_retrieval():
    result = RAGRetrieverFactory.create(
        backend="qdrant",
        semantic_retriever=_semantic_retriever(),
        lexical_retriever=_lexical_retriever(),
    )

    assert isinstance(result, HybridRetriever)
    assert result.candidate_k == 5


def test_hybrid_backend_requires_lexical_retriever():
    with pytest.raises(ValueError, match="PostgreSQL lexical retriever"):
        RAGRetrieverFactory.create(
            backend="postgres",
            semantic_retriever=_semantic_retriever(),
        )


def test_cross_encoder_wraps_hybrid_retriever(monkeypatch):
    import rag.retrieval.factory as factory_module

    monkeypatch.setattr(factory_module, "CrossEncoderReranker", FakeReranker)

    result = RAGRetrieverFactory.create(
        backend="postgres",
        semantic_retriever=_semantic_retriever(),
        lexical_retriever=_lexical_retriever(),
        reranker="cross_encoder",
        reranker_model_id="custom-model",
        reranker_onnx_filename="custom.onnx",
        reranker_max_length=4096,
        reranker_candidate_k=20,
    )

    assert isinstance(result, RerankingRetriever)
    assert isinstance(result.retriever, HybridRetriever)
    assert isinstance(result.reranker, FakeReranker)

    assert result.candidate_k == 20
    assert result.retriever.candidate_k == 20

    assert result.reranker.model_id == "custom-model"
    assert result.reranker.onnx_filename == "custom.onnx"
    assert result.reranker.max_length == 4096


@pytest.mark.parametrize(
    "reranker",
    ["unsupported", "", "token_overlap"],
)
def test_rejects_unsupported_reranker(reranker):
    with pytest.raises(ValueError, match="Unsupported RAG reranker"):
        RAGRetrieverFactory.create(
            backend="postgres",
            semantic_retriever=_semantic_retriever(),
            lexical_retriever=_lexical_retriever(),
            reranker=reranker,
        )


@pytest.mark.parametrize("backend", ["", "mysql", "snowflake", "unknown"])
def test_rejects_unsupported_backend(backend):
    with pytest.raises(ValueError, match="Unsupported RAG retrieval backend"):
        RAGRetrieverFactory.create(
            backend=backend,
            semantic_retriever=_semantic_retriever(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reranker_max_length", 0),
        ("reranker_max_length", -1),
        ("reranker_candidate_k", 0),
        ("reranker_candidate_k", -1),
    ],
)
def test_rejects_invalid_reranker_configuration(field, value):
    kwargs = {
        "backend": "postgres",
        "semantic_retriever": _semantic_retriever(),
        "lexical_retriever": _lexical_retriever(),
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        RAGRetrieverFactory.create(**kwargs)
