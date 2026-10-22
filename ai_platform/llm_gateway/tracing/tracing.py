"""
OpenTelemetry tracing configuration.

This module configures distributed tracing for the LLM Gateway.

Responsibilities
----------------
* Configure the OpenTelemetry SDK.
* Configure the OTLP exporter.
* Configure batch span processing.
* Instrument the FastAPI application.

No application business logic belongs in this module.
"""

import logging
import os
import re
from urllib.parse import urlsplit
from weakref import WeakSet

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import (
    FastAPIInstrumentor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace import Event
from opentelemetry.trace import Link, Status, StatusCode
from opentelemetry.sdk.trace.export import BatchSpanProcessor


_TRUE_VALUES = {"true", "1", "yes", "on"}
_NOOP_TRACER_PROVIDER = trace.NoOpTracerProvider()
_GATEWAY_TRACER_PROVIDER = _NOOP_TRACER_PROVIDER


def gateway_tracing_enabled() -> bool:
    """Return whether gateway tracing export is enabled."""
    return os.getenv("OTEL_EXPORT_ENABLED", "false").strip().lower() in _TRUE_VALUES


class _GatewayTracerProxy:
    """Resolve the tracer on each operation so export configuration is respected."""

    def __init__(self, instrumentation_name: str):
        self._instrumentation_name = instrumentation_name

    def _get_tracer(self):
        # Provider selection is made during application startup. Do not
        # change tracing behavior if the environment changes mid-process.
        return _GATEWAY_TRACER_PROVIDER.get_tracer(self._instrumentation_name)

    def start_as_current_span(self, *args, **kwargs):
        return self._get_tracer().start_as_current_span(*args, **kwargs)

    def start_span(self, *args, **kwargs):
        return self._get_tracer().start_span(*args, **kwargs)


def get_gateway_tracer(instrumentation_name: str):
    """Return a tracer proxy that respects OTEL_EXPORT_ENABLED."""
    return _GatewayTracerProxy(instrumentation_name)


tracer = get_gateway_tracer(__name__)

logger = logging.getLogger(__name__)


# These attributes can contain credentials, query parameters, user content,
# or caller-controlled correlation values. Keep exported telemetry operational.
_SENSITIVE_ATTRIBUTE_NAMES = {
    "http.target",
    "http.url",
    "url.full",
    "url.original",
    "url.query",
    "url.fragment",
    "validation.proof_id",
    "exception.message",
    "exception.stacktrace",
    "input",
    "output",
    "content",
    "prompt",
    "messages",
}


# Export only attributes deliberately used for operational gateway tracing.
# Unknown semantic-convention and application attributes are denied by default.
_ALLOWED_ATTRIBUTE_NAMES = {
    "llm.model",
    "llm.requested_provider",
    "llm.provider",
    "llm.attempt_count",
    "provider.name",
    "provider.model",
    "provider.stream.error",
}


def _safe_attribute(key, value):
    if not isinstance(key, str) or key not in _ALLOWED_ATTRIBUTE_NAMES:
        return False

    if key in {
        "llm.model",
        "llm.requested_provider",
        "llm.provider",
        "provider.name",
        "provider.model",
    }:
        return (
            isinstance(value, str)
            and 0 < len(value) <= 128
            and re.fullmatch(r"[A-Za-z0-9_./:-]+", value) is not None
        )

    if key == "llm.attempt_count":
        return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1000

    if key == "provider.stream.error":
        return isinstance(value, bool)

    return False


def _path_from_value(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlsplit(
            value
            if "://" in value
            else "//placeholder" + (value if value.startswith("/") else "/" + value)
        )
        return parsed.path or "/"
    except (TypeError, ValueError):
        return None


def _is_metrics_span(span):
    attrs = getattr(span, "attributes", None) or {}

    # Route is the preferred stable attribute. Other URL fields are fallback
    # sources for older instrumentation versions and untemplated requests.
    candidates = [
        attrs.get("http.route"),
        attrs.get("http.target"),
        attrs.get("url.path"),
        attrs.get("url.full"),
        attrs.get("url.original"),
        attrs.get("http.url"),
        getattr(span, "name", None),
    ]

    for candidate in candidates:
        path = _path_from_value(candidate)
        if path and path.rstrip("/").lower() == "/metrics":
            return True
    return False


def _sanitize_attributes(attributes):
    return {key: value for key, value in (attributes or {}).items() if _safe_attribute(key, value)}


def _sanitize_events(events):
    sanitized = []
    for event in events or ():
        event_name = getattr(event, "name", "")
        # Event names can be caller-controlled too. Preserve only the
        # conventional exception name; normalize other names.
        safe_name = (
            "exception"
            if isinstance(event_name, str) and event_name.lower() == "exception"
            else "event"
        )
        sanitized.append(
            Event(
                name=safe_name,
                attributes=_sanitize_attributes(getattr(event, "attributes", None)),
                timestamp=getattr(event, "timestamp", None),
            )
        )
    return tuple(sanitized)


def _sanitize_links(links):
    sanitized = []
    for link in links or ():
        context = getattr(link, "context", None)
        if context is None:
            continue
        sanitized.append(
            Link(
                context=context,
                attributes=_sanitize_attributes(getattr(link, "attributes", None)),
            )
        )
    return tuple(sanitized)


def _sanitize_resource(resource):
    """Keep only explicitly approved service metadata in exported spans."""
    if resource is None:
        return None

    from opentelemetry.sdk.resources import Resource

    attributes = getattr(resource, "attributes", {}) or {}
    allowed = {}
    for key in ("service.name", "service.version", "deployment.environment"):
        value = attributes.get(key)
        if (
            isinstance(value, str)
            and 0 < len(value) <= 128
            and re.fullmatch(r"[A-Za-z0-9_.:/-]+", value) is not None
        ):
            allowed[key] = value

    return Resource(allowed, schema_url=getattr(resource, "schema_url", None))


def _make_safe_span(span):
    """Create a fresh SDK ReadableSpan with sensitive telemetry removed."""
    original_name = getattr(span, "name", "") or ""
    internal_names = {
        "gateway.chat",
        "gateway.embeddings",
        "gateway.stream",
        "provider_call",
        "provider_health_check",
        "provider_selection",
        "response_parsing",
        "authentication",
    }

    # Normalize HTTP instrumentation names and discard arbitrary operation names.
    if original_name in internal_names:
        name = original_name
    elif re.match(
        r"^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s",
        original_name,
        re.IGNORECASE,
    ):
        name = "http.server"
    else:
        name = "operation"

    original_status = getattr(span, "status", None)
    status_code = getattr(original_status, "status_code", StatusCode.UNSET)
    # Status descriptions frequently contain exception messages or user data.
    safe_status = Status(status_code=status_code, description=None)

    return ReadableSpan(
        name=name,
        context=getattr(span, "context", None),
        parent=getattr(span, "parent", None),
        resource=_sanitize_resource(getattr(span, "resource", None)),
        attributes=_sanitize_attributes(getattr(span, "attributes", None)),
        events=_sanitize_events(getattr(span, "events", ())),
        links=_sanitize_links(getattr(span, "links", ())),
        kind=getattr(span, "kind", None),
        instrumentation_info=getattr(span, "instrumentation_info", None),
        status=safe_status,
        start_time=getattr(span, "start_time", None),
        end_time=getattr(span, "end_time", None),
        instrumentation_scope=getattr(span, "instrumentation_scope", None),
    )


class SpanFilterProcessor(SpanProcessor):
    """Filter unwanted spans and sanitize attributes before export."""

    def __init__(self, next_processor: SpanProcessor):
        self.next_processor = next_processor

    def on_start(self, span, parent_context=None):
        # Preserve the SpanProcessor lifecycle contract. The downstream
        # BatchSpanProcessor handles on_start without queuing the span.
        if self.next_processor:
            self.next_processor.on_start(span, parent_context)

    def on_end(self, span):
        if span is None:
            return

        name = (getattr(span, "name", "") or "").lower()
        if name.endswith("http receive") or name.endswith("http send"):
            return

        if _is_metrics_span(span):
            return

        if self.next_processor:
            self.next_processor.on_end(_make_safe_span(span))

    def shutdown(self):
        if self.next_processor:
            self.next_processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if self.next_processor:
            return self.next_processor.force_flush(timeout_millis)
        return True


def server_request_hook(span, scope):
    """Keep the hook deliberately content-free; never record proof headers."""
    if not (span and hasattr(span, "is_recording") and span.is_recording()):
        return
    # Intentionally do not parse or attach x-otel-proof-id. If a future
    # validation flow needs it, handle it outside exported telemetry.
    return


def configure_tracing(app: FastAPI) -> None:
    """
    Configure OpenTelemetry tracing.

    This function:

    * Creates a resource describing the service.
    * Configures a TracerProvider.
    * Registers an OTLP exporter.
    * Adds a BatchSpanProcessor wrapped in a SpanFilterProcessor.
    * Instruments the FastAPI application.
    """
    global _GATEWAY_TRACER_PROVIDER

    service_name = os.getenv(
        "OTEL_SERVICE_NAME",
        "llm-gateway",
    )

    service_version = os.getenv(
        "OTEL_SERVICE_VERSION",
        "1.0.0",
    )

    deployment_environment = os.getenv(
        "OTEL_DEPLOYMENT_ENVIRONMENT",
        "development",
    )

    export_enabled = gateway_tracing_enabled()

    # Resource values are deployment configuration, not request data.
    # Bound and normalize them so malformed environment values cannot inject
    # unbounded or arbitrary content into exported telemetry.
    def _safe_resource_value(value, fallback):
        if not isinstance(value, str):
            return fallback
        value = value.strip()
        if not value or len(value) > 128:
            return fallback
        if re.fullmatch(r"[A-Za-z0-9_.:/-]+", value) is None:
            return fallback
        return value

    service_name = _safe_resource_value(service_name, "llm-gateway")
    service_version = _safe_resource_value(service_version, "unknown")
    deployment_environment = _safe_resource_value(deployment_environment, "unknown")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": deployment_environment,
        }
    )

    # Keep instrumentation idempotent for app reloads and repeated test setup.
    configured_apps = globals().setdefault("_CONFIGURED_TRACING_APPS", WeakSet())
    if app in configured_apps:
        return

    # The gateway owns its provider; never attach its exporter to the
    # process-wide provider, which may be used by unrelated components.
    tracer_provider = _NOOP_TRACER_PROVIDER

    if export_enabled:
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if not otlp_endpoint:
            raise RuntimeError("OTEL_EXPORT_ENABLED=true requires OTEL_EXPORTER_OTLP_ENDPOINT.")

        # Create one gateway provider/exporter per process. Repeated app
        # configuration reuses this provider rather than adding exporters.
        if _GATEWAY_TRACER_PROVIDER is _NOOP_TRACER_PROVIDER:
            gateway_provider = TracerProvider(resource=resource)
            exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "false").strip().lower()
                in {"true", "1", "yes", "on"},
            )
            batch_processor = BatchSpanProcessor(
                exporter,
                schedule_delay_millis=5000,
                max_queue_size=2048,
                max_export_batch_size=512,
            )
            gateway_provider.add_span_processor(SpanFilterProcessor(batch_processor))
            _GATEWAY_TRACER_PROVIDER = gateway_provider

        tracer_provider = _GATEWAY_TRACER_PROVIDER

    FastAPIInstrumentor.instrument_app(
        app,
        tracer_provider=tracer_provider,
        excluded_urls=r"^/metrics$",
        exclude_spans=["send", "receive"],
        server_request_hook=server_request_hook,
    )
    configured_apps.add(app)
