FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd \
    --create-home \
    --uid 10001 \
    appuser

COPY pyproject.toml .
COPY ai_platform ./ai_platform

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        fastapi \
        uvicorn \
        pydantic \
        pydantic-settings \
        httpx \
        prometheus-client \
        opentelemetry-api \
        opentelemetry-sdk \
        openai \
        opentelemetry-exporter-otlp-proto-grpc \
        opentelemetry-instrumentation-fastapi

USER appuser

EXPOSE 8000

CMD ["uvicorn", "ai_platform.llm_gateway.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
