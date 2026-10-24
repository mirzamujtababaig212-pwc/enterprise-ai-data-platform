from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from ai_platform.llm_gateway.tracing.tracing import SpanFilterProcessor


@dataclass
class FakeSpan:
    name: str
    attributes: dict
    events: tuple = ()
    links: tuple = ()
    status: object = None


class RecordingProcessor:
    def __init__(self):
        self.started = []
        self.ended = []
        self.shutdown_called = False

    def on_start(self, span, parent_context=None):
        self.started.append(span)

    def on_end(self, span):
        self.ended.append(span)

    def shutdown(self):
        self.shutdown_called = True

    def force_flush(self, timeout_millis=30000):
        return True


@pytest.mark.parametrize(
    "name,attributes",
    [
        ("GET http receive", {}),
        ("GET http send", {}),
        ("GET /metrics", {"http.target": "/metrics"}),
        ("GET /metrics", {"http.target": "/metrics?probe=1"}),
        ("GET /metrics", {"url.path": "/METRICS"}),
        (
            "GET /metrics",
            {"url.full": "http://localhost:8000/metrics?probe=1"},
        ),
    ],
)
def test_span_filter_drops_transport_and_metrics_spans(name, attributes):
    downstream = RecordingProcessor()
    processor = SpanFilterProcessor(downstream)

    span = FakeSpan(name=name, attributes=attributes)
    processor.on_end(span)

    assert downstream.ended == []


def test_span_filter_forwards_ordinary_spans():
    downstream = RecordingProcessor()
    processor = SpanFilterProcessor(downstream)
    span = FakeSpan(
        name="POST /v1/chat/completions",
        attributes={"http.route": "/v1/chat/completions"},
    )

    processor.on_end(span)

    assert len(downstream.ended) == 1
    assert downstream.ended[0].name == "http.server"


def test_span_filter_does_not_export_sensitive_url_or_proof_attributes():
    downstream = RecordingProcessor()
    processor = SpanFilterProcessor(downstream)
    span = FakeSpan(
        name="POST /v1/chat/completions",
        attributes={
            "http.route": "/v1/chat/completions",
            "url.full": "https://example.test/path?secret=privacy-canary-url",
            "url.query": "secret=privacy-canary-url",
            "http.target": "/path?secret=privacy-canary-url",
            "http.url": "https://example.test/path?secret=privacy-canary-url",
            "validation.proof_id": "privacy-canary-proof",
        },
    )

    processor.on_end(span)

    assert len(downstream.ended) == 1
    exported = downstream.ended[0].attributes
    assert "http.route" not in exported
    assert not any(
        key.startswith("url.")
        or key
        in {
            "http.target",
            "http.url",
            "validation.proof_id",
        }
        for key in exported
    )
    assert "privacy-canary-url" not in repr(exported)
    assert "privacy-canary-proof" not in repr(exported)


def test_server_request_hook_does_not_log_or_export_proof_id():
    from ai_platform.llm_gateway.tracing.tracing import server_request_hook

    span = Mock()
    span.is_recording.return_value = True
    scope = {
        "headers": [
            (b"x-otel-proof-id", b"privacy-canary-proof"),
        ],
    }

    with __import__("unittest").mock.patch(
        "ai_platform.llm_gateway.tracing.tracing.logger"
    ) as logger:
        server_request_hook(span, scope)

    logger.info.assert_not_called()
    span.set_attribute.assert_not_called()


def test_span_filter_forwards_lifecycle_and_flush_operations():
    downstream = RecordingProcessor()
    processor = SpanFilterProcessor(downstream)
    span = FakeSpan(name="GET /health", attributes={})

    processor.on_start(span)
    assert downstream.started == [span]
    assert processor.force_flush() is True

    processor.shutdown()
    assert downstream.shutdown_called


def test_span_filter_sanitizes_event_link_and_status_content():
    from opentelemetry.sdk.trace import Event
    from opentelemetry.trace import (
        Link,
        SpanContext,
        Status,
        StatusCode,
        TraceFlags,
    )

    from ai_platform.llm_gateway.tracing.tracing import SpanFilterProcessor

    event_canary = "privacy-canary-event-secret"
    link_canary = "privacy-canary-link-secret"
    status_canary = "privacy-canary-status-secret"

    context = SpanContext(
        trace_id=1,
        span_id=1,
        is_remote=False,
        trace_flags=TraceFlags(1),
    )

    span = FakeSpan(
        name="POST /v1/chat/completions",
        attributes={"http.route": "/v1/chat/completions"},
        events=(
            Event(
                name="exception",
                attributes={
                    "exception.message": event_canary,
                    "exception.stacktrace": event_canary,
                },
                timestamp=123,
            ),
            Event(
                name=event_canary,
                attributes={"provider.name": "safe-provider"},
                timestamp=124,
            ),
        ),
        links=(
            Link(
                context=context,
                attributes={
                    "url.full": link_canary,
                    "x-otel-proof-id": link_canary,
                    "provider.name": "safe-link-provider",
                },
            ),
        ),
        status=Status(
            status_code=StatusCode.ERROR,
            description=status_canary,
        ),
    )

    downstream = RecordingProcessor()
    SpanFilterProcessor(downstream).on_end(span)

    assert len(downstream.ended) == 1
    exported = downstream.ended[0]

    assert exported.status.status_code is StatusCode.ERROR
    assert exported.status.description is None

    assert len(exported.events) == 2
    assert exported.events[0].name == "exception"
    assert exported.events[0].attributes == {}
    assert exported.events[1].name == "event"
    assert exported.events[1].attributes == {"provider.name": "safe-provider"}

    assert len(exported.links) == 1
    assert exported.links[0].context == context
    assert exported.links[0].attributes == {"provider.name": "safe-link-provider"}

    exported_repr = repr(exported)
    assert event_canary not in exported_repr
    assert link_canary not in exported_repr
    assert status_canary not in exported_repr


def test_gateway_tracer_proxy_uses_noop_when_export_is_disabled(monkeypatch):
    from opentelemetry import trace

    import ai_platform.llm_gateway.tracing.tracing as tracing_module

    monkeypatch.setenv("OTEL_EXPORT_ENABLED", "false")
    monkeypatch.setattr(
        tracing_module,
        "_GATEWAY_TRACER_PROVIDER",
        tracing_module._NOOP_TRACER_PROVIDER,
    )

    def unexpected_global_tracer(*args, **kwargs):
        raise AssertionError("Disabled tracing must not use the global provider")

    monkeypatch.setattr(trace, "get_tracer", unexpected_global_tracer)
    span = tracing_module.get_gateway_tracer("tests.gateway.disabled").start_span("disabled-span")

    assert span.is_recording() is False


def test_gateway_tracer_proxy_uses_configured_provider_if_environment_changes(monkeypatch):
    import ai_platform.llm_gateway.tracing.tracing as tracing_module

    expected_tracer = Mock()
    expected_span = Mock()
    expected_tracer.start_span.return_value = expected_span
    gateway_provider = Mock()
    gateway_provider.get_tracer.return_value = expected_tracer

    monkeypatch.setenv("OTEL_EXPORT_ENABLED", "true")
    monkeypatch.setattr(tracing_module, "_GATEWAY_TRACER_PROVIDER", gateway_provider)

    # Once configured, the provider must remain authoritative even if
    # the environment variable changes while this process is running.
    monkeypatch.setenv("OTEL_EXPORT_ENABLED", "false")

    actual_span = tracing_module.get_gateway_tracer("tests.gateway.enabled").start_span(
        "enabled-span"
    )

    assert actual_span is expected_span
    gateway_provider.get_tracer.assert_called_once_with("tests.gateway.enabled")
    expected_tracer.start_span.assert_called_once_with("enabled-span")


def test_configure_tracing_explicitly_uses_noop_provider_when_disabled(monkeypatch):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from opentelemetry import trace

    import ai_platform.llm_gateway.tracing.tracing as tracing_module

    instrument_app = Mock()
    monkeypatch.setattr(
        tracing_module,
        "FastAPIInstrumentor",
        SimpleNamespace(instrument_app=instrument_app),
    )
    monkeypatch.setenv("OTEL_EXPORT_ENABLED", "false")

    app = FastAPI()
    tracing_module.configure_tracing(app)

    assert instrument_app.call_count == 1
    kwargs = instrument_app.call_args.kwargs
    assert isinstance(kwargs["tracer_provider"], trace.NoOpTracerProvider)
    assert kwargs["excluded_urls"] == r"^/metrics$"
    assert kwargs["exclude_spans"] == ["send", "receive"]


def test_configure_tracing_does_not_reuse_or_replace_global_provider(monkeypatch):
    from types import SimpleNamespace

    from fastapi import FastAPI
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    import ai_platform.llm_gateway.tracing.tracing as tracing_module

    global_provider = TracerProvider()
    instrument_app = Mock()
    set_global_provider = Mock()
    get_global_provider = Mock(return_value=global_provider)
    exporter = Mock()

    monkeypatch.setenv("OTEL_EXPORT_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setattr(
        tracing_module,
        "_GATEWAY_TRACER_PROVIDER",
        tracing_module._NOOP_TRACER_PROVIDER,
    )
    monkeypatch.setattr(
        tracing_module,
        "FastAPIInstrumentor",
        SimpleNamespace(instrument_app=instrument_app),
    )
    monkeypatch.setattr(
        tracing_module,
        "OTLPSpanExporter",
        Mock(return_value=exporter),
    )
    monkeypatch.setattr(trace, "set_tracer_provider", set_global_provider)
    monkeypatch.setattr(trace, "get_tracer_provider", get_global_provider)

    app = FastAPI()
    tracing_module.configure_tracing(app)

    assert instrument_app.call_count == 1
    gateway_provider = instrument_app.call_args.kwargs["tracer_provider"]
    assert isinstance(gateway_provider, TracerProvider)
    assert gateway_provider is not global_provider
    assert tracing_module._GATEWAY_TRACER_PROVIDER is gateway_provider
    set_global_provider.assert_not_called()
    get_global_provider.assert_not_called()

    # Reconfiguring the same app must not instrument it a second time.
    tracing_module.configure_tracing(app)
    assert instrument_app.call_count == 1
    assert exporter.shutdown.call_count == 0


def test_span_filter_drops_unknown_attributes_and_sensitive_values():
    downstream = RecordingProcessor()
    span = FakeSpan(
        name="POST /v1/chat/completions/user-123",
        attributes={
            "http.route": "/v1/chat/completions",
            "llm.model": "model-prod",
            "provider.name": "provider-a",
            "tenant.id": "privacy-canary-tenant",
            "authorization": "privacy-canary-token",
            "llm.model.secret": "privacy-canary-model-secret",
            "llm.requested_provider": "provider?token=privacy-canary-query",
        },
    )

    SpanFilterProcessor(downstream).on_end(span)

    assert len(downstream.ended) == 1
    exported = downstream.ended[0]
    assert exported.name == "http.server"
    assert exported.attributes == {
        "llm.model": "model-prod",
        "provider.name": "provider-a",
    }
    rendered = repr(exported)
    assert "privacy-canary" not in rendered
    assert "user-123" not in rendered


def test_span_filter_normalizes_unknown_span_names():
    downstream = RecordingProcessor()
    span = FakeSpan(
        name="custom-operation privacy-canary-user",
        attributes={"tenant.id": "privacy-canary-tenant"},
    )

    SpanFilterProcessor(downstream).on_end(span)

    assert downstream.ended[0].name == "operation"
    assert downstream.ended[0].attributes == {}
    assert "privacy-canary" not in repr(downstream.ended[0])


def test_span_filter_drops_raw_route_attributes():
    downstream = RecordingProcessor()
    span = FakeSpan(
        name="POST /v1/customers/customer-7391/chat",
        attributes={
            "http.route": "/v1/customers/customer-7391/chat",
            "llm.model": "model-prod",
        },
    )

    SpanFilterProcessor(downstream).on_end(span)

    assert len(downstream.ended) == 1
    exported = downstream.ended[0]
    assert exported.name == "http.server"
    assert exported.attributes == {"llm.model": "model-prod"}
    assert "customer-7391" not in repr(exported)


def test_span_filter_sanitizes_resource_attributes():
    from opentelemetry.sdk.resources import Resource

    downstream = RecordingProcessor()
    span = FakeSpan(name="gateway.chat", attributes={})
    span.resource = Resource(
        {
            "service.name": "llm-gateway",
            "service.version": "1.0.0",
            "deployment.environment": "production",
            "process.command_line": "privacy-canary-resource",
            "host.name": "privacy-canary-host",
            "customer.id": "privacy-canary-customer",
        }
    )

    SpanFilterProcessor(downstream).on_end(span)

    assert len(downstream.ended) == 1
    exported_resource = downstream.ended[0].resource
    assert exported_resource.attributes == {
        "service.name": "llm-gateway",
        "service.version": "1.0.0",
        "deployment.environment": "production",
    }
    assert "privacy-canary" not in repr(exported_resource.attributes)
