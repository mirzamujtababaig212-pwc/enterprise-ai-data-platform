from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_seconds: float
    reason: str


RETRYABLE_STATUS_CODES = {
    408,
    409,
    425,
    429,
    500,
    502,
    503,
    504,
}


def should_retry(
    *,
    status_code: int | None,
    attempt: int,
    max_attempts: int,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    jitter: bool = True,
) -> RetryDecision:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be greater than zero")

    if attempt < 0:
        raise ValueError("attempt must not be negative")

    if base_delay < 0:
        raise ValueError("base_delay must not be negative")

    if max_delay < 0:
        raise ValueError("max_delay must not be negative")

    if attempt >= max_attempts - 1:
        return RetryDecision(
            retry=False,
            delay_seconds=0.0,
            reason="maximum retry attempts reached",
        )

    if status_code not in RETRYABLE_STATUS_CODES:
        return RetryDecision(
            retry=False,
            delay_seconds=0.0,
            reason="status code is not retryable",
        )

    exponential_delay = base_delay * (2**attempt)

    jitter_seconds = random.uniform(0.0, 0.25) if jitter else 0.0

    delay = min(
        exponential_delay + jitter_seconds,
        max_delay,
    )

    return RetryDecision(
        retry=True,
        delay_seconds=delay,
        reason=f"retryable status code: {status_code}",
    )
