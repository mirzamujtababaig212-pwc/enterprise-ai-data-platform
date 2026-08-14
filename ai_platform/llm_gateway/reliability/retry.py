from __future__ import annotations

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
) -> RetryDecision:
    if attempt >= max_attempts:
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

    delay = min(2**attempt, 30)

    return RetryDecision(
        retry=True,
        delay_seconds=float(delay),
        reason=f"retryable status code: {status_code}",
    )
