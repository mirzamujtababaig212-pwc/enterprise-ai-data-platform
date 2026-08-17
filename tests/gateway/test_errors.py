from ai_platform.llm_gateway.exceptions import (
    AuthenticationError,
    CapabilityNotSupportedError,
    RateLimitError,
)
from ai_platform.llm_gateway.models.errors import GatewayErrorCode


def test_rate_limit_error_is_retryable():
    error = RateLimitError(
        "provider rate limit exceeded",
        provider="openai",
        model="gpt-4.1-mini",
        retry_after_seconds=10,
    )

    result = error.to_error()

    assert result.code == GatewayErrorCode.RATE_LIMITED
    assert result.retryable is True
    assert result.provider == "openai"
    assert result.model == "gpt-4.1-mini"
    assert result.retry_after_seconds == 10


def test_authentication_error_is_not_retryable():
    error = AuthenticationError(
        "invalid provider credentials",
        provider="openai",
    )

    result = error.to_error()

    assert result.code == GatewayErrorCode.AUTHENTICATION_FAILED
    assert result.retryable is False


def test_capability_error_is_not_retryable():
    error = CapabilityNotSupportedError(
        "model does not support embeddings",
        provider="openai",
        model="gpt-4.1-mini",
    )

    result = error.to_error()

    assert result.code == GatewayErrorCode.CAPABILITY_NOT_SUPPORTED
    assert result.retryable is False
