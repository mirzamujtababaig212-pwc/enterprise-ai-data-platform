from datetime import datetime
from uuid import UUID

from ai_platform.llm_gateway.models.usage import UsageMetrics


def test_usage_metrics():

    metrics = UsageMetrics(
        provider="openai",
        model="gpt-4",
        prompt_tokens=12,
        completion_tokens=8,
        total_tokens=20,
        latency_ms=105.4,
        cost=0.0012,
    )

    assert metrics.provider == "openai"
    assert metrics.model == "gpt-4"

    assert metrics.prompt_tokens == 12
    assert metrics.completion_tokens == 8
    assert metrics.total_tokens == 20

    assert metrics.latency_ms == 105.4
    assert metrics.cost == 0.0012


def test_request_id_is_uuid():

    metrics = UsageMetrics(
        provider="openai",
        model="gpt-4",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        latency_ms=10,
        cost=0.0,
    )

    UUID(metrics.request_id)


def test_timestamp_generated():

    metrics = UsageMetrics(
        provider="openai",
        model="gpt-4",
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        latency_ms=10,
        cost=0.0,
    )

    assert isinstance(metrics.timestamp, datetime)


def test_model_dump():

    metrics = UsageMetrics(
        provider="openai",
        model="gpt-4",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=50.5,
        cost=0.01,
    )

    data = metrics.model_dump()

    assert data["provider"] == "openai"
    assert data["model"] == "gpt-4"
    assert data["prompt_tokens"] == 10
    assert data["completion_tokens"] == 5
    assert data["total_tokens"] == 15
    assert data["latency_ms"] == 50.5
    assert data["cost"] == 0.01

    UUID(data["request_id"])
    assert isinstance(data["timestamp"], datetime)
