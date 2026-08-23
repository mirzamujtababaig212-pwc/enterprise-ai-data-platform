from decimal import Decimal

from ai_platform.llm_gateway.gateway.cost import (
    build_usage_record,
    calculate_cost,
)


def test_calculate_gpt_41_mini_cost():
    cost = calculate_cost(
        model="gpt-4.1-mini",
        prompt_tokens=1_000,
        completion_tokens=500,
    )

    expected = Decimal("1000") / Decimal("1000000") * Decimal("0.40") + Decimal("500") / Decimal(
        "1000000"
    ) * Decimal("1.60")

    assert cost == expected


def test_unknown_model_has_zero_cost():
    cost = calculate_cost(
        model="unknown-model",
        prompt_tokens=1_000,
        completion_tokens=500,
    )

    assert cost == Decimal("0")


def test_usage_record():
    record = build_usage_record(
        provider="openai",
        model="gpt-4.1-mini",
        prompt_tokens=1_000,
        completion_tokens=500,
        request_id="req_123",
    )

    assert record.provider == "openai"
    assert record.model == "gpt-4.1-mini"
    assert record.prompt_tokens == 1_000
    assert record.completion_tokens == 500
    assert record.total_tokens == 1_500
    assert record.request_id == "req_123"
