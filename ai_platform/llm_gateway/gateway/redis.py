from __future__ import annotations

import os

from redis.asyncio import Redis


def create_redis_client() -> Redis:
    redis_url = os.getenv(
        "REDIS_URL",
        "redis://localhost:6379/0",
    )

    return Redis.from_url(
        redis_url,
        decode_responses=True,
    )
