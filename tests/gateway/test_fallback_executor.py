"""Tests for provider fallback execution."""

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
    executor = FallbackExecutor()

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
    executor = FallbackExecutor()

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
