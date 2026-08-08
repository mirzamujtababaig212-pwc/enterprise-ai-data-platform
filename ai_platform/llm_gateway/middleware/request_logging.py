import time
import json
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

from ai_platform.llm_gateway.logging.logger import get_logger
from ai_platform.llm_gateway.security.redaction import (
    sanitize_body,
    sanitize_headers,
)

from ai_platform.llm_gateway.observability.prometheus import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    PROVIDER_REQUESTS_TOTAL,
    MODEL_REQUESTS_TOTAL,
    INPUT_TOKENS_TOTAL,
    OUTPUT_TOKENS_TOTAL,
    ESTIMATED_COST_TOTAL,
    ERRORS_TOTAL,
)

logger = get_logger()


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        logger.info("Request logging middleware executed")

        start = time.perf_counter()

        body = await request.body()

        payload = {}
        if body:
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                payload = {}

        sanitized_payload = sanitize_body(payload)

        provider = payload.get("provider")
        model = payload.get("model")

        response = await call_next(request)

        status_code = str(response.status_code)
        metrics = getattr(
            request.state,
            "metrics",
            {},
        )
        tokens_in = metrics.get("tokens_in")

        tokens_out = metrics.get("tokens_out")

        estimated_cost = metrics.get("estimated_cost")

        latency_ms = int((time.perf_counter() - start) * 1000)
        method = request.method
        endpoint = request.url.path

        client_ip = request.client.host if request.client else "unknown"

        user_agent = request.headers.get(
            "user-agent",
            "unknown",
        )

        headers = dict(request.headers)
        sanitized_headers = sanitize_headers(headers)

        request_id = getattr(
            request.state,
            "request_id",
            None,
        )

        extra = {
            "request_id": request_id,
            "method": method,
            "endpoint": endpoint,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "headers": sanitized_headers,
            "request_body": sanitized_payload,
        }

        if provider is not None:
            extra["provider"] = provider

        if model is not None:
            extra["model"] = model

        if tokens_in is not None:
            extra["tokens_in"] = tokens_in

        if tokens_out is not None:
            extra["tokens_out"] = tokens_out

        if estimated_cost is not None:
            extra["estimated_cost"] = estimated_cost

        logger.info(
            "HTTP Request Completed",
            extra=extra,
        )

        #
        # Prometheus Metrics
        #

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            endpoint=endpoint,
        ).observe(latency_ms / 1000)

        if provider:

            PROVIDER_REQUESTS_TOTAL.labels(
                provider=provider,
            ).inc()

        if model:

            MODEL_REQUESTS_TOTAL.labels(
                model=model,
            ).inc()

        if tokens_in is not None:

            INPUT_TOKENS_TOTAL.inc(tokens_in)

        if tokens_out is not None:

            OUTPUT_TOKENS_TOTAL.inc(tokens_out)

        if estimated_cost is not None:

            ESTIMATED_COST_TOTAL.inc(estimated_cost)

        if response.status_code >= 400:

            ERRORS_TOTAL.labels(
                status_code=status_code,
            ).inc()

        return response
