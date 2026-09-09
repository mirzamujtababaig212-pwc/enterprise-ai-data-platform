"""Provider fallback execution."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ai_platform.llm_gateway.metrics.prometheus import (
    FALLBACK_REQUESTS_TOTAL,
    PROVIDER_ERRORS_TOTAL,
    PROVIDER_LATENCY_SECONDS,
    PROVIDER_REQUESTS_TOTAL,
    PROVIDER_RETRIES_TOTAL,
)
from ai_platform.llm_gateway.reliability.failure_classifier import (
    FailureCategory,
    ProviderFailureClassifier,
    failure_classifier,
)
from ai_platform.llm_gateway.reliability.retry import should_retry


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
    model_name: str | None = None


class FallbackExecutor:
    """Execute providers in order until one succeeds."""

    def __init__(
        self,
        classifier: ProviderFailureClassifier | None = None,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")

        if base_delay < 0:
            raise ValueError("base_delay must not be negative")

        if max_delay < 0:
            raise ValueError("max_delay must not be negative")

        self.classifier = classifier or failure_classifier
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

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

        primary_provider_name = self._provider_name(providers[0])

        for index, provider in enumerate(providers):
            provider_name = self._provider_name(provider)

            if index > 0:
                FALLBACK_REQUESTS_TOTAL.labels(
                    primary_provider=primary_provider_name,
                    fallback_provider=provider_name,
                ).inc()

            for attempt in range(self.max_retries + 1):
                started_at = time.perf_counter()

                try:
                    PROVIDER_REQUESTS_TOTAL.labels(
                        provider=provider_name,
                    ).inc()

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

                    PROVIDER_ERRORS_TOTAL.labels(
                        provider=provider_name,
                        error_type=category.value,
                    ).inc()

                    if self.classifier.is_retryable(error):
                        status_code = getattr(error, "status_code", None)

                        decision = should_retry(
                            status_code=status_code,
                            attempt=attempt,
                            max_attempts=self.max_retries + 1,
                            base_delay=self.base_delay,
                            max_delay=self.max_delay,
                        )

                        # The classifier is authoritative for exception types
                        # such as TimeoutError and ConnectionError that do not
                        # expose an HTTP status code.
                        retry = (
                            decision.retry
                            if status_code is not None
                            else attempt < self.max_retries
                        )

                        if retry:
                            PROVIDER_RETRIES_TOTAL.labels(
                                provider=provider_name,
                                failure_category=category.value,
                            ).inc()

                            if status_code is not None:
                                delay = decision.delay_seconds
                            else:
                                delay = min(
                                    self.base_delay * (2**attempt),
                                    self.max_delay,
                                )

                            if delay > 0:
                                await asyncio.sleep(delay)

                            continue

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

                    break

                finally:
                    PROVIDER_LATENCY_SECONDS.labels(
                        provider=provider_name,
                    ).observe(time.perf_counter() - started_at)

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
