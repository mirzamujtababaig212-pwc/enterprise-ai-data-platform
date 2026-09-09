"""Tests for provider fallback execution."""

from unittest.mock import AsyncMock, patch

import pytest
from prometheus_client import REGISTRY

from ai_platform.llm_gateway.routing.fallback_executor import (
    FallbackExecutor,
)


class FakeProvider:
    """Simple provider used by fallback tests."""

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.asyncio
async def test_fallback_moves_to_second_provider() -> None:
    executor = FallbackExecutor(base_delay=0)

    first = FakeProvider("openai")
    second = FakeProvider("gemini")

    async def call(provider: FakeProvider):
        if provider.name == "openai":
            raise TimeoutError("provider timeout")

        return {
            "provider": provider.name,
            "response": "success",
        }

    result = await executor.execute(
        [first, second],
        call,
    )

    assert result.provider_name == "gemini"
    assert result.response["response"] == "success"

    assert len(result.attempts) == 2

    assert result.attempts[0].provider_name == "openai"
    assert not result.attempts[0].success

    assert result.attempts[1].provider_name == "gemini"
    assert result.attempts[1].success


@pytest.mark.asyncio
async def test_successful_first_provider_does_not_call_second() -> None:
    executor = FallbackExecutor()

    first = FakeProvider("openai")
    second = FakeProvider("gemini")

    calls: list[str] = []

    async def call(provider: FakeProvider):
        calls.append(provider.name)

        return {
            "provider": provider.name,
        }

    result = await executor.execute(
        [first, second],
        call,
    )

    assert result.provider_name == "openai"
    assert calls == ["openai"]
    assert len(result.attempts) == 1


@pytest.mark.asyncio
async def test_non_retryable_error_stops_fallback() -> None:
    executor = FallbackExecutor()

    first = FakeProvider("openai")
    second = FakeProvider("gemini")

    calls: list[str] = []

    async def call(provider: FakeProvider):
        calls.append(provider.name)

        raise RuntimeError("authentication failed")

    with pytest.raises(RuntimeError, match="authentication failed"):
        await executor.execute(
            [first, second],
            call,
        )

    assert calls == ["openai"]


@pytest.mark.asyncio
async def test_all_retryable_providers_fail() -> None:
    executor = FallbackExecutor(base_delay=0)

    first = FakeProvider("openai")
    second = FakeProvider("gemini")

    async def call(provider: FakeProvider):
        raise TimeoutError(
            f"{provider.name} timeout",
        )

    with pytest.raises(TimeoutError, match="gemini timeout"):
        await executor.execute(
            [first, second],
            call,
        )


def _fallback_metric_value(
    primary_provider: str,
    fallback_provider: str,
) -> float:
    for metric in REGISTRY.collect():
        if metric.name != "llm_gateway_fallback_requests":
            continue

        for sample in metric.samples:
            if (
                sample.name == "llm_gateway_fallback_requests_total"
                and sample.labels.get("primary_provider") == primary_provider
                and sample.labels.get("fallback_provider") == fallback_provider
            ):
                return sample.value

    return 0.0


@pytest.mark.asyncio
async def test_fallback_metric_increases_when_second_provider_is_attempted():
    executor = FallbackExecutor()

    first = FakeProvider("metric-openai")
    second = FakeProvider("metric-gemini")

    before = _fallback_metric_value(
        "metric-openai",
        "metric-gemini",
    )

    async def call(provider: FakeProvider):
        if provider.name == "metric-openai":
            raise TimeoutError("provider timeout")

        return {
            "provider": provider.name,
            "response": "success",
        }

    result = await executor.execute(
        [first, second],
        call,
    )

    assert result.provider_name == "metric-gemini"

    after = _fallback_metric_value(
        "metric-openai",
        "metric-gemini",
    )

    assert after == before + 1


def _provider_request_metric_value(provider: str) -> float:
    for metric in REGISTRY.collect():
        if metric.name != "llm_gateway_provider_requests":
            continue

        for sample in metric.samples:
            if (
                sample.name == "llm_gateway_provider_requests_total"
                and sample.labels.get("provider") == provider
            ):
                return sample.value

    return 0.0


@pytest.mark.asyncio
async def test_provider_request_metric_counts_each_provider_attempt():
    executor = FallbackExecutor(
        max_retries=0,
        base_delay=0,
    )

    first = FakeProvider("requests-openai")
    second = FakeProvider("requests-ollama")

    before_first = _provider_request_metric_value(first.name)
    before_second = _provider_request_metric_value(second.name)

    async def call(provider: FakeProvider):
        if provider.name == first.name:
            raise TimeoutError("provider timeout")

        return {
            "provider": provider.name,
            "response": "success",
        }

    result = await executor.execute(
        [first, second],
        call,
    )

    assert result.provider_name == second.name

    after_first = _provider_request_metric_value(first.name)
    after_second = _provider_request_metric_value(second.name)

    assert after_first == before_first + 1
    assert after_second == before_second + 1


@pytest.mark.asyncio
async def test_retry_succeeds_on_same_provider_before_fallback():
    executor = FallbackExecutor(
        max_retries=2,
        base_delay=0,
    )

    first = FakeProvider("retry-openai")
    second = FakeProvider("retry-gemini")
    calls: list[str] = []

    async def call(provider: FakeProvider):
        calls.append(provider.name)

        if len(calls) == 1:
            raise TimeoutError("provider timeout")

        return {
            "provider": provider.name,
            "response": "success",
        }

    result = await executor.execute(
        [first, second],
        call,
    )

    assert result.provider_name == "retry-openai"
    assert calls == ["retry-openai", "retry-openai"]
    assert len(result.attempts) == 1
    assert result.attempts[0].success is True


@pytest.mark.asyncio
async def test_retry_exhaustion_then_falls_back():
    executor = FallbackExecutor(
        max_retries=2,
        base_delay=0,
    )

    first = FakeProvider("exhaust-openai")
    second = FakeProvider("exhaust-gemini")
    calls: list[str] = []

    async def call(provider: FakeProvider):
        calls.append(provider.name)

        if provider.name == "exhaust-openai":
            raise TimeoutError("provider timeout")

        return {
            "provider": provider.name,
            "response": "fallback success",
        }

    result = await executor.execute(
        [first, second],
        call,
    )

    assert result.provider_name == "exhaust-gemini"
    assert calls == [
        "exhaust-openai",
        "exhaust-openai",
        "exhaust-openai",
        "exhaust-gemini",
    ]

    assert len(result.attempts) == 2
    assert result.attempts[0].success is False
    assert result.attempts[1].success is True


@pytest.mark.asyncio
async def test_quota_failure_skips_retry_and_falls_back():
    executor = FallbackExecutor(
        max_retries=2,
        base_delay=0,
    )

    first = FakeProvider("quota-openai")
    second = FakeProvider("quota-gemini")
    calls: list[str] = []

    async def call(provider: FakeProvider):
        calls.append(provider.name)

        if provider.name == "quota-openai":
            raise RuntimeError("insufficient_quota")

        return {
            "provider": provider.name,
            "response": "quota fallback success",
        }

    result = await executor.execute(
        [first, second],
        call,
    )

    assert result.provider_name == "quota-gemini"
    assert calls == [
        "quota-openai",
        "quota-gemini",
    ]


@pytest.mark.asyncio
async def test_retryable_exception_uses_backoff_without_real_sleep():
    executor = FallbackExecutor(
        max_retries=1,
        base_delay=2.0,
        max_delay=8.0,
    )

    first = FakeProvider("backoff-openai")
    calls = 0

    async def call(provider: FakeProvider):
        nonlocal calls
        calls += 1

        if calls == 1:
            raise TimeoutError("provider timeout")

        return {
            "provider": provider.name,
            "response": "success",
        }

    with patch(
        "ai_platform.llm_gateway.routing.fallback_executor.asyncio.sleep",
        new_callable=AsyncMock,
    ) as sleep:
        result = await executor.execute(
            [first],
            call,
        )

    assert result.provider_name == "backoff-openai"
    sleep.assert_awaited_once_with(2.0)


def _retry_metric_value(
    provider: str,
    failure_category: str,
) -> float:
    for metric in REGISTRY.collect():
        if metric.name != "llm_gateway_provider_retries":
            continue

        for sample in metric.samples:
            if (
                sample.name == "llm_gateway_provider_retries_total"
                and sample.labels.get("provider") == provider
                and sample.labels.get("failure_category") == failure_category
            ):
                return sample.value

    return 0.0


@pytest.mark.asyncio
async def test_retry_metric_increases_for_retry_attempt():
    executor = FallbackExecutor(
        max_retries=1,
        base_delay=0,
    )

    first = FakeProvider("metric-retry-openai")

    before = _retry_metric_value(
        "metric-retry-openai",
        "timeout",
    )

    calls = 0

    async def call(provider: FakeProvider):
        nonlocal calls
        calls += 1

        if calls == 1:
            raise TimeoutError("provider timeout")

        return {
            "provider": provider.name,
            "response": "success",
        }

    result = await executor.execute(
        [first],
        call,
    )

    assert result.provider_name == "metric-retry-openai"

    after = _retry_metric_value(
        "metric-retry-openai",
        "timeout",
    )

    assert after == before + 1


def _provider_latency_count(provider: str) -> float:
    for metric in REGISTRY.collect():
        if metric.name != "llm_gateway_provider_latency_seconds":
            continue

        for sample in metric.samples:
            if (
                sample.name == "llm_gateway_provider_latency_seconds_count"
                and sample.labels.get("provider") == provider
            ):
                return sample.value

    return 0.0


def _provider_error_metric_value(
    provider: str,
    error_type: str,
) -> float:
    for metric in REGISTRY.collect():
        if metric.name != "llm_gateway_provider_errors":
            continue

        for sample in metric.samples:
            if (
                sample.name == "llm_gateway_provider_errors_total"
                and sample.labels.get("provider") == provider
                and sample.labels.get("error_type") == error_type
            ):
                return sample.value

    return 0.0


@pytest.mark.asyncio
async def test_provider_latency_metric_records_successful_attempt():
    executor = FallbackExecutor(base_delay=0)

    provider = FakeProvider("telemetry-success-openai")

    before = _provider_latency_count(provider.name)

    async def call(_provider: FakeProvider):
        return {"response": "success"}

    result = await executor.execute([provider], call)

    assert result.provider_name == provider.name

    after = _provider_latency_count(provider.name)

    assert after == before + 1


@pytest.mark.asyncio
async def test_provider_error_metric_records_failed_attempt():
    executor = FallbackExecutor(
        max_retries=0,
        base_delay=0,
    )

    provider = FakeProvider("telemetry-error-openai")

    before_errors = _provider_error_metric_value(
        provider.name,
        "timeout",
    )
    before_latency = _provider_latency_count(provider.name)

    async def call(_provider: FakeProvider):
        raise TimeoutError("provider timeout")

    with pytest.raises(TimeoutError, match="provider timeout"):
        await executor.execute([provider], call)

    after_errors = _provider_error_metric_value(
        provider.name,
        "timeout",
    )
    after_latency = _provider_latency_count(provider.name)

    assert after_errors == before_errors + 1
    assert after_latency == before_latency + 1


@pytest.mark.asyncio
async def test_provider_metrics_record_each_retry_attempt():
    executor = FallbackExecutor(
        max_retries=2,
        base_delay=0,
    )

    provider = FakeProvider("telemetry-retry-openai")

    before_errors = _provider_error_metric_value(
        provider.name,
        "timeout",
    )
    before_latency = _provider_latency_count(provider.name)

    calls = 0

    async def call(_provider: FakeProvider):
        nonlocal calls
        calls += 1

        if calls <= 2:
            raise TimeoutError("provider timeout")

        return {"response": "success"}

    result = await executor.execute([provider], call)

    assert result.provider_name == provider.name
    assert calls == 3

    after_errors = _provider_error_metric_value(
        provider.name,
        "timeout",
    )
    after_latency = _provider_latency_count(provider.name)

    assert after_errors == before_errors + 2
    assert after_latency == before_latency + 3
