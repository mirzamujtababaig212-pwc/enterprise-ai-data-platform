from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: Decimal
    output_per_1m: Decimal


MODEL_PRICING: dict[str, ModelPricing] = {
    "gpt-4.1-mini": ModelPricing(
        input_per_1m=Decimal("0.40"),
        output_per_1m=Decimal("1.60"),
    ),
    "gpt-4.1": ModelPricing(
        input_per_1m=Decimal("2.00"),
        output_per_1m=Decimal("8.00"),
    ),
}


@dataclass(frozen=True)
class UsageRecord:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: Decimal
    request_id: str


def calculate_cost(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Decimal:
    pricing = MODEL_PRICING.get(model)

    if pricing is None:
        return Decimal("0")

    input_cost = Decimal(prompt_tokens) / Decimal("1000000") * pricing.input_per_1m

    output_cost = Decimal(completion_tokens) / Decimal("1000000") * pricing.output_per_1m

    return input_cost + output_cost


def build_usage_record(
    *,
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    request_id: str,
) -> UsageRecord:
    total_tokens = prompt_tokens + completion_tokens

    estimated_cost = calculate_cost(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    return UsageRecord(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost,
        request_id=request_id,
    )
