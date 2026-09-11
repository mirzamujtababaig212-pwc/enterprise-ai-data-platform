from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from rag.models import DocumentChunk, EmbeddedChunk
from rag.observability import (
    RAG_RETRIEVAL_EMPTY_RESULTS_TOTAL,
    RAG_RETRIEVAL_ERRORS_TOTAL,
    RAG_RETRIEVAL_REQUESTS_TOTAL,
)
from rag.retrieval.retriever import SemanticRetriever
from rag.stores.in_memory import InMemoryVectorStore


class FakeEmbeddingService:
    async def embed(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0, 0.0)


class FailingEmbeddingService:
    async def embed(self, text: str) -> tuple[float, ...]:
        raise RuntimeError("embedding failure")


def _metric_value(metric) -> float:
    return sum(
        sample.value for sample in metric.collect()[0].samples if sample.name.endswith("_total")
    )


def _make_tracer() -> tuple[trace.Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("tests.rag.retrieval")

    return tracer, exporter


def _make_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.mark.asyncio
async def test_retrieval_increments_request_metric() -> None:
    store = _make_store()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="chunk-1",
                    document_id="doc-1",
                    content="Electric vehicle information.",
                ),
                embedding=(1.0, 0.0, 0.0),
            )
        ]
    )

    tracer, _ = _make_tracer()
    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        tracer=tracer,
    )

    before = _metric_value(RAG_RETRIEVAL_REQUESTS_TOTAL)

    results = await retriever.retrieve(
        "electric vehicle",
        top_k=1,
    )

    after = _metric_value(RAG_RETRIEVAL_REQUESTS_TOTAL)

    assert len(results) == 1
    assert after == before + 1


@pytest.mark.asyncio
async def test_empty_retrieval_increments_empty_results_metric() -> None:
    store = _make_store()

    tracer, _ = _make_tracer()
    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        tracer=tracer,
    )

    before = _metric_value(RAG_RETRIEVAL_EMPTY_RESULTS_TOTAL)

    results = await retriever.retrieve(
        "electric vehicle",
        top_k=1,
    )

    after = _metric_value(RAG_RETRIEVAL_EMPTY_RESULTS_TOTAL)

    assert results == []
    assert after == before + 1


@pytest.mark.asyncio
async def test_retrieval_error_increments_error_metric() -> None:
    store = _make_store()

    tracer, _ = _make_tracer()
    retriever = SemanticRetriever(
        embedding_service=FailingEmbeddingService(),
        vector_store=store,
        tracer=tracer,
    )

    before = _metric_value(RAG_RETRIEVAL_ERRORS_TOTAL)

    with pytest.raises(RuntimeError, match="embedding failure"):
        await retriever.retrieve(
            "electric vehicle",
            top_k=1,
        )

    after = _metric_value(RAG_RETRIEVAL_ERRORS_TOTAL)

    assert after == before + 1


@pytest.mark.asyncio
async def test_retrieval_span_records_quality_attributes() -> None:
    store = _make_store()

    await store.upsert(
        [
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="chunk-1",
                    document_id="doc-1",
                    content="Electric vehicle information.",
                ),
                embedding=(1.0, 0.0, 0.0),
            ),
            EmbeddedChunk(
                chunk=DocumentChunk(
                    id="chunk-2",
                    document_id="doc-1",
                    content="Related vehicle information.",
                ),
                embedding=(0.8, 0.6, 0.0),
            ),
        ]
    )

    tracer, exporter = _make_tracer()
    retriever = SemanticRetriever(
        embedding_service=FakeEmbeddingService(),
        vector_store=store,
        tracer=tracer,
    )

    results = await retriever.retrieve(
        "electric vehicle",
        top_k=2,
        min_score=0.5,
    )

    assert len(results) == 2

    spans = [span for span in exporter.get_finished_spans() if span.name == "rag.retrieval"]

    assert len(spans) == 1

    span = spans[0]

    assert span.status.status_code is trace.StatusCode.UNSET
    assert span.attributes["rag.retrieval.top_k"] == 2
    assert span.attributes["rag.retrieval.min_score"] == 0.5
    assert span.attributes["rag.retrieval.candidate_count"] == 2
    assert span.attributes["rag.retrieval.returned_count"] == 2
    assert span.attributes["rag.retrieval.score_min"] == pytest.approx(0.8)
    assert span.attributes["rag.retrieval.score_max"] == pytest.approx(1.0)
    assert span.attributes["rag.retrieval.score_avg"] == pytest.approx(0.9)
    assert span.attributes["rag.retrieval.metadata_filter_applied"] is False
    assert span.attributes["rag.retrieval.governance_policy_applied"] is False
