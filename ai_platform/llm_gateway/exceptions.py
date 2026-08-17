from __future__ import annotations

from typing import Any

from ai_platform.llm_gateway.models.errors import GatewayErrorCode


class LLMGatewayError(Exception):
    """Base exception for all LLM Gateway failures."""

    code: GatewayErrorCode = GatewayErrorCode.INTERNAL_ERROR
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        retry_after_seconds: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.provider = provider
        self.model = model
        self.retry_after_seconds = retry_after_seconds
        self.details = details or {}

    def to_error(self):
        from ai_platform.llm_gateway.models.errors import GatewayError

        return GatewayError(
            code=self.code,
            message=self.message,
            provider=self.provider,
            model=self.model,
            retryable=self.retryable,
            retry_after_seconds=self.retry_after_seconds,
            details=self.details,
        )


class AuthenticationError(LLMGatewayError):
    code = GatewayErrorCode.AUTHENTICATION_FAILED


class AuthorizationError(LLMGatewayError):
    code = GatewayErrorCode.AUTHORIZATION_FAILED


class RateLimitError(LLMGatewayError):
    code = GatewayErrorCode.RATE_LIMITED
    retryable = True


class ProviderUnavailableError(LLMGatewayError):
    code = GatewayErrorCode.PROVIDER_UNAVAILABLE
    retryable = True


class ModelNotFoundError(LLMGatewayError):
    code = GatewayErrorCode.MODEL_NOT_FOUND


class InvalidRequestError(LLMGatewayError):
    code = GatewayErrorCode.INVALID_REQUEST


class CapabilityNotSupportedError(LLMGatewayError):
    code = GatewayErrorCode.CAPABILITY_NOT_SUPPORTED


class ProviderError(LLMGatewayError):
    code = GatewayErrorCode.PROVIDER_ERROR


class InternalGatewayError(LLMGatewayError):
    code = GatewayErrorCode.INTERNAL_ERROR
