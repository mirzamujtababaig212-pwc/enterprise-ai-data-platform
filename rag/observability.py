from __future__ import annotations

from prometheus_client import Counter, Histogram

from opentelemetry import trace

RAG_RETRIEVAL_REQUESTS_TOTAL = Counter(
    "deldai_rag_retrieval_requests_total",
    "Total RAG retrieval requests.",
)

RAG_RETRIEVAL_EMPTY_RESULTS_TOTAL = Counter(
    "deldai_rag_retrieval_empty_results_total",
    "Total RAG retrieval requests that returned no results.",
)

RAG_RETRIEVAL_ERRORS_TOTAL = Counter(
    "deldai_rag_retrieval_errors_total",
    "Total RAG retrieval requests that failed.",
)

RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    "deldai_rag_retrieval_duration_seconds",
    "RAG retrieval latency in seconds.",
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
    ),
)

RAG_RETRIEVAL_TRACER = trace.get_tracer(
    "deldai.rag",
)
