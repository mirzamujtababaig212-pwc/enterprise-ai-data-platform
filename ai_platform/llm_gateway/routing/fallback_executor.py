"""Provider fallback execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ai_platform.llm_gateway.reliability.failure_classifier import (
    FailureCategory,
    ProviderFailureClassifier,
    failure_classifier,
)


@dataclass(frozen=True)
class ProviderAttempt:
    """Result of one provider execution attempt."""

    provider_name: str
    success: bool
    failure_category: FailureCategory | None = None
    error: Exception | None = None


@dataclass(frozen=True)
class FallbackResult:
    """Final result returned by fallback execution."""

    response: Any
    provider_name: str
    attempts: tuple[ProviderAttempt, ...]


class FallbackExecutor:
    """Execute providers in order until one succeeds."""

    def __init__(
        self,
        classifier: ProviderFailureClassifier | None = None,
        max_retries=2,
        base_delay=1.0,
        max_delay=8.0,
    ) -> None:
        self.classifier = classifier or failure_classifier

    async def execute(
        self,
        providers: Sequence[Any],
        call: Callable[[Any], Awaitable[Any]],
    ) -> FallbackResult:
        """Execute the provider sequence with fallback."""

        if not providers:
            raise ValueError("At least one provider is required")

        attempts: list[ProviderAttempt] = []

        last_error: Exception | None = None

        for provider in providers:
            provider_name = self._provider_name(provider)

            try:
                response = await call(provider)

                attempts.append(
                    ProviderAttempt(
                        provider_name=provider_name,
                        success=True,
                    )
                )

                return FallbackResult(
                    response=response,
                    provider_name=provider_name,
                    attempts=tuple(attempts),
                )

            except Exception as error:
                last_error = error

                category = self.classifier.classify(error)

                attempts.append(
                    ProviderAttempt(
                        provider_name=provider_name,
                        success=False,
                        failure_category=category,
                        error=error,
                    )
                )

                if not self.classifier.is_fallback_eligible(error):
                    raise

        if last_error is not None:
            raise last_error

        raise RuntimeError("Provider fallback execution failed")

    @staticmethod
    def _provider_name(provider: Any) -> str:
        """Extract a stable provider name."""

        for attribute in (
            "name",
            "provider_name",
        ):
            value = getattr(provider, attribute, None)

            if value:
                return str(value)

        return provider.__class__.__name__
