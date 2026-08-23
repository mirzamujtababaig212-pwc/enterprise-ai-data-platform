"""Backward-compatible retry helpers for the gateway layer.

The canonical retry policy lives in:
    ai_platform.llm_gateway.reliability.retry
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ai_platform.llm_gateway.reliability.retry import (
    RETRYABLE_STATUS_CODES,
    should_retry,
)


async def retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int = 2,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
) -> T:
    """
    Backward-compatible async retry wrapper.

    Retry decisions are delegated to the canonical reliability
    retry policy.
    """

    if max_retries < 0:
        raise ValueError("max_retries must not be negative")

    if base_delay < 0:
        raise ValueError("base_delay must not be negative")

    if max_delay < 0:
        raise ValueError("max_delay must not be negative")

    max_attempts = max_retries + 1

    for attempt in range(max_attempts):
        try:
            return await operation()

        except Exception as exc:
            status_code = getattr(exc, "status_code", None)

            decision = should_retry(
                status_code=status_code,
                attempt=attempt,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
            )

            if not decision.retry:
                raise

            await asyncio.sleep(decision.delay_seconds)

    raise RuntimeError("Retry execution failed unexpectedly")


__all__ = [
    "RETRYABLE_STATUS_CODES",
    "retry_async",
    "should_retry",
]
