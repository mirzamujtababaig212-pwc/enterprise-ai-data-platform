from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
) -> T:

    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):

        try:
            return await operation()

        except Exception as exc:
            last_exception = exc

            if attempt >= max_attempts:
                break

            delay = base_delay_seconds * (2 ** (attempt - 1))

            logger.warning(
                "Provider request failed; " "retrying attempt=%s/%s delay=%.2f error=%s",
                attempt,
                max_attempts,
                delay,
                exc,
            )

            await asyncio.sleep(delay)

    assert last_exception is not None

    raise last_exception
