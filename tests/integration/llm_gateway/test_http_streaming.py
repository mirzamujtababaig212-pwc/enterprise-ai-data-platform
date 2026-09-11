from unittest.mock import patch

from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from ai_platform.llm_gateway.api.main import app
from ai_platform.llm_gateway.metrics.prometheus import (
    PROVIDER_LATENCY_SECONDS,
    PROVIDER_REQUESTS_TOTAL,
)
from ai_platform.llm_gateway.routing.router import Router
from ai_platform.llm_gateway.routing.fallback_executor import FallbackExecutor

from tests.gateway.test_router_fallback_integration import (
    FakeCapabilityService,
    FakeProvider,
    FakeRoutingResolver,
)

client = TestClient(app)

HEADERS = {"x-api-key": "super-secret-key"}


def _counter_value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


def _histogram_count(histogram, **labels) -> float:
    target_labels = {key: str(value) for key, value in labels.items()}

    for metric in histogram.collect():
        for sample in metric.samples:
            if sample.name.endswith("_count") and dict(sample.labels) == target_labels:
                return sample.value

    return 0.0


async def fake_gemini_stream(request):
    yield "RAG retrieves relevant context."
    yield " The LLM uses that context to answer the question."


def test_http_streaming():
    with patch(
        "ai_platform.llm_gateway.api.main.router.route_stream",
        new=fake_gemini_stream,
    ):
        response = client.post(
            "/v1/chat",
            json={
                "prompt": "Explain RAG in one sentence.",
                "provider": "gemini",
                "model": "gemini-chat",
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": True,
            },
            headers=HEADERS,
        )

    assert response.status_code == 200

    content_type = response.headers.get(
        "content-type",
        "",
    )

    assert content_type.startswith("text/event-stream")

    body = response.text

    assert "data: RAG retrieves relevant context.\n\n" in body
    assert "data:  The LLM uses that context to answer the question.\n\n" in body
    assert "data: [DONE]\n\n" in body

    first_chunk = "data: RAG retrieves relevant context.\n\n"
    second_chunk = "data:  The LLM uses that context to answer the question.\n\n"
    done = "data: [DONE]\n\n"

    assert body.index(first_chunk) < body.index(second_chunk)
    assert body.index(second_chunk) < body.index(done)


def test_http_streaming_propagates_gateway_provider_observability(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(exporter),
    )
    tracer = tracer_provider.get_tracer(
        "test-http-streaming-observability",
    )

    monkeypatch.setattr(
        "ai_platform.llm_gateway.api.main.router",
        Router(
            routing_resolver=FakeRoutingResolver(
                [
                    FakeProvider(
                        "provider-http-stream",
                        stream_chunks=[
                            "chunk-1",
                            "chunk-2",
                        ],
                    ),
                ],
            ),
            fallback_executor=FallbackExecutor(),
        ),
    )

    monkeypatch.setattr(
        "ai_platform.llm_gateway.routing.router.capability_service",
        FakeCapabilityService(),
    )

    monkeypatch.setattr(
        "ai_platform.llm_gateway.routing.router.tracer",
        tracer,
    )

    provider_name = "provider-http-stream"

    before_requests = _counter_value(
        PROVIDER_REQUESTS_TOTAL,
        provider=provider_name,
    )

    before_latency_count = _histogram_count(
        PROVIDER_LATENCY_SECONDS,
        provider=provider_name,
    )

    with tracer.start_as_current_span("test.http.request") as parent_span:
        response = client.post(
            "/v1/chat",
            json={
                "prompt": "Explain RAG in one sentence.",
                "provider": provider_name,
                "model": "test-stream-model",
                "temperature": 0.7,
                "max_tokens": 100,
                "stream": True,
            },
            headers=HEADERS,
        )

        parent_trace_id = parent_span.get_span_context().trace_id

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "text/event-stream",
    )

    body = response.text

    assert "data: chunk-1\n\n" in body
    assert "data: chunk-2\n\n" in body
    assert "data: [DONE]\n\n" in body

    after_requests = _counter_value(
        PROVIDER_REQUESTS_TOTAL,
        provider=provider_name,
    )

    after_latency_count = _histogram_count(
        PROVIDER_LATENCY_SECONDS,
        provider=provider_name,
    )

    assert after_requests == before_requests + 1
    assert after_latency_count == before_latency_count + 1

    spans = exporter.get_finished_spans()

    gateway_span = next(span for span in spans if span.name == "gateway.stream")

    provider_span = next(span for span in spans if span.name == "provider_call")

    assert gateway_span.get_span_context().trace_id == parent_trace_id
    assert provider_span.get_span_context().trace_id == parent_trace_id

    assert provider_span.parent.span_id == gateway_span.get_span_context().span_id

    assert gateway_span.status.status_code is trace.StatusCode.OK
    assert provider_span.status.status_code is trace.StatusCode.OK

    assert gateway_span.attributes["llm.model"] == "test-stream-model"
    assert gateway_span.attributes["llm.requested_provider"] == provider_name

    assert provider_span.attributes["provider.name"] == provider_name
    assert provider_span.attributes["provider.model"] == "test-stream-model"

    for span in (gateway_span, provider_span):
        assert "prompt" not in span.attributes
        assert "response" not in span.attributes
        assert "user.id" not in span.attributes
        assert "session.id" not in span.attributes
