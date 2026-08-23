from __future__ import annotations

import time
from dataclasses import dataclass

from redis.asyncio import Redis


class RateLimitExceeded(Exception):
    """Raised when a tenant exceeds a configured rate limit."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int,
    ) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class RateLimitConfig:
    requests_per_minute: int = 60
    tokens_per_minute: int = 100_000


class RedisRateLimiter:
    """
    Distributed fixed-window rate limiter.

    Keys are tenant-scoped:

        ratelimit:{tenant_id}:{minute}

    Separate counters are maintained for requests and tokens.
    """

    def __init__(
        self,
        redis: Redis,
        config: RateLimitConfig,
    ) -> None:
        self.redis = redis
        self.config = config

    @staticmethod
    def _window() -> int:
        return int(time.time() // 60)

    async def check_request(
        self,
        *,
        tenant_id: str,
        requested_tokens: int = 0,
    ) -> None:
        window = self._window()

        request_key = f"ratelimit:{tenant_id}:{window}:requests"
        token_key = f"ratelimit:{tenant_id}:{window}:tokens"

        pipe = self.redis.pipeline()

        pipe.incr(request_key)

        if requested_tokens > 0:
            pipe.incrby(token_key, requested_tokens)

        pipe.expire(request_key, 120)

        if requested_tokens > 0:
            pipe.expire(token_key, 120)

        results = await pipe.execute()

        request_count = int(results[0])

        token_count = 0

        if requested_tokens > 0:
            token_count = int(results[1])

        if request_count > self.config.requests_per_minute:
            raise RateLimitExceeded(
                "Request rate limit exceeded.",
                retry_after_seconds=60,
            )

        if requested_tokens > 0 and token_count > self.config.tokens_per_minute:
            raise RateLimitExceeded(
                "Token rate limit exceeded.",
                retry_after_seconds=60,
            )
