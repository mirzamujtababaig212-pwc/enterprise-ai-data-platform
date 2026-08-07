import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request


class RequestIDMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        request_id = str(uuid.uuid4())
        from opentelemetry import trace

        span = trace.get_current_span()
        span_context = span.get_span_context()
        trace_id = None
        if span_context.is_valid:
            trace_id = format(
                span_context.trace_id,
                "032x",
            )
        request.state.trace_id = trace_id
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
