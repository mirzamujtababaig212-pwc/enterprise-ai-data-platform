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

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.fastapi import (
    FastAPIInstrumentor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

tracer = trace.get_tracer(__name__)

logger = logging.getLogger(__name__)


class SpanFilterProcessor(SpanProcessor):
    def __init__(self, next_processor: SpanProcessor):
        self.next_processor = next_processor

    def on_start(self, span, parent_context=None):
        if self.next_processor:
            self.next_processor.on_start(span, parent_context)

    def on_end(self, span):
        if not span:
            return
        attrs = getattr(span, "attributes", None) or {}
        span_name = getattr(span, "name", "") or ""
        if span_name.endswith("http receive") or span_name.endswith("http send"):
            return

        if attrs.get("http.target") == "/metrics" or attrs.get("url.path") == "/metrics":
            return

        if self.next_processor:
            self.next_processor.on_end(span)

    def shutdown(self):
        if self.next_processor:
            self.next_processor.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        if self.next_processor:
            return self.next_processor.force_flush(timeout_millis)
        return True


def server_request_hook(span, scope):
    logger.info(
        "OTEL server request hook executed", extra={"span_name": getattr(span, "name", None)}
    )
    if not (span and hasattr(span, "is_recording") and span.is_recording()):
        return

    proof_id = None

    if isinstance(scope, dict):
        headers = scope.get("headers", [])
        for item in headers:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item

                key_str = k.decode("utf-8", errors="ignore") if isinstance(k, bytes) else str(k)

                if key_str.lower() == "x-otel-proof-id":
                    proof_id = (
                        v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
                    )
                    break

    logger.info(
        "OTEL server request hook executed",
        extra={
            "proof_id": proof_id,
            "span_name": getattr(span, "name", None),
        },
    )

    if proof_id:
        span.set_attribute("validation.proof_id", proof_id)


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

    export_enabled = os.getenv(
        "OTEL_EXPORT_ENABLED",
        "false",
    ).strip().lower() in {
        "true",
        "1",
        "yes",
        "on",
    }

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": deployment_environment,
        }
    )

    tracer_provider = TracerProvider(
        resource=resource,
    )

    trace.set_tracer_provider(
        tracer_provider,
    )

    if export_enabled:
        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
        )

        if not otlp_endpoint:
            raise RuntimeError("OTEL_EXPORT_ENABLED=true requires OTEL_EXPORTER_OTLP_ENDPOINT.")

        exporter = OTLPSpanExporter(
            endpoint=otlp_endpoint,
            insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "false").lower()
            in {"true", "1", "yes", "on"},
        )

        batch_processor = BatchSpanProcessor(
            exporter,
            schedule_delay_millis=500,  # Flush every 500ms instead of 5000ms
            max_export_batch_size=1,
        )
        filtered_processor = SpanFilterProcessor(batch_processor)

        tracer_provider.add_span_processor(filtered_processor)

    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls=r"^/metrics$",
        exclude_spans=["send", "receive"],
        server_request_hook=server_request_hook,
    )
