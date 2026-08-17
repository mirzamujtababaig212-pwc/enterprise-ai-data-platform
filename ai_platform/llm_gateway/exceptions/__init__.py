from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    GatewayException,
    ProviderNotFound,
    ProviderTimeout,
    ProviderUnavailable,
)
from ai_platform.llm_gateway.exceptions.gateway_exceptions import (
    InvalidRequest as LegacyInvalidRequest,
)
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from ai_platform.llm_gateway.exceptions.provider_exceptions import (
    ProviderError as ProviderException,
)
from ai_platform.llm_gateway.models.errors import GatewayErrorCode


class LLMGatewayError(Exception):
    """Base exception for errors exposed by the LLM Gateway."""

    code: GatewayErrorCode = GatewayErrorCode.INTERNAL_ERROR
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        retry_after_seconds: float | None = None,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.retry_after_seconds = retry_after_seconds
        self.details = details or {}

    def to_error_response(self):
        return self.to_error()

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
    """Authentication with a gateway/provider failed."""

    code = GatewayErrorCode.AUTHENTICATION_FAILED


class AuthorizationError(LLMGatewayError):
    """The caller is authenticated but not authorized."""

    code = GatewayErrorCode.AUTHORIZATION_FAILED


class RateLimitError(LLMGatewayError):
    """The gateway or provider rate limit was exceeded."""

    code = GatewayErrorCode.RATE_LIMITED
    retryable = True


class ProviderUnavailableError(LLMGatewayError):
    """The selected provider is unavailable."""

    code = GatewayErrorCode.PROVIDER_UNAVAILABLE
    retryable = True


class ModelNotFoundError(LLMGatewayError):
    """The requested model does not exist or is not registered."""

    code = GatewayErrorCode.MODEL_NOT_FOUND


class InvalidRequestError(LLMGatewayError):
    """The gateway request is invalid."""

    code = GatewayErrorCode.INVALID_REQUEST


class CapabilityNotSupportedError(LLMGatewayError):
    """The requested capability is not supported."""

    code = GatewayErrorCode.CAPABILITY_NOT_SUPPORTED


class ProviderError(LLMGatewayError):
    """Generic provider failure."""

    code = GatewayErrorCode.PROVIDER_ERROR


class InternalGatewayError(LLMGatewayError):
    """Unexpected internal gateway failure."""

    code = GatewayErrorCode.INTERNAL_ERROR


__all__ = [
    "LLMGatewayError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "ProviderUnavailableError",
    "ModelNotFoundError",
    "InvalidRequestError",
    "CapabilityNotSupportedError",
    "ProviderError",
    "InternalGatewayError",
    "GatewayException",
    "ProviderNotFound",
    "ProviderTimeout",
    "ProviderUnavailable",
    "LegacyInvalidRequest",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderConnectionError",
    "ProviderException",
]
