from ai_platform.llm_gateway.gateway.rate_limit import (
    RateLimitConfig,
)


def test_rate_limit_defaults():
    config = RateLimitConfig()

    assert config.requests_per_minute == 60
    assert config.tokens_per_minute == 100_000


def test_rate_limit_custom_configuration():
    config = RateLimitConfig(
        requests_per_minute=10,
        tokens_per_minute=5_000,
    )

    assert config.requests_per_minute == 10
    assert config.tokens_per_minute == 5_000
