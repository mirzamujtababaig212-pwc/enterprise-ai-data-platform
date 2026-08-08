from datetime import datetime, timezone
from uuid import UUID

from ai_platform.llm_gateway.models.usage import UsageMetrics


def test_usage_metrics():
    metrics = UsageMetrics(
        latency_ms=105.4,
        tokens_in=12,
        tokens_out=8,
        estimated_cost=0.0012,
        status="success",
    )

    assert metrics.latency_ms == 105.4
    assert metrics.tokens_in == 12
    assert metrics.tokens_out == 8
    assert metrics.estimated_cost == 0.0012
    assert metrics.status == "success"


def test_request_id_is_uuid():
    metrics = UsageMetrics(
        tokens_in=1,
        tokens_out=1,
        estimated_cost=0.0,
    )

    UUID(metrics.request_id)


def test_timestamp_generated():
    metrics = UsageMetrics(
        tokens_in=1,
        tokens_out=1,
        estimated_cost=0.0,
    )

    assert isinstance(metrics.timestamp, datetime)
    assert metrics.timestamp.tzinfo is not None
    assert metrics.timestamp.utcoffset() == timezone.utc.utcoffset(metrics.timestamp)


def test_model_dump():
    metrics = UsageMetrics(
        latency_ms=50.5,
        tokens_in=10,
        tokens_out=5,
        estimated_cost=0.01,
        status="success",
    )

    data = metrics.model_dump()

    assert data["latency_ms"] == 50.5
    assert data["tokens_in"] == 10
    assert data["tokens_out"] == 5
    assert data["estimated_cost"] == 0.01
    assert data["status"] == "success"

    UUID(data["request_id"])
    assert isinstance(data["timestamp"], datetime)
