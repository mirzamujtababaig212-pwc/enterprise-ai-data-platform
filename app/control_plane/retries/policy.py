import random
from collections.abc import Callable

from pydantic import BaseModel, Field, model_validator

from app.control_plane.retries.models import FailureDisposition


class RetryDecision(BaseModel):
    allowed: bool
    attempt: int
    max_attempts: int
    delay_seconds: float
    reason: str


class RetryPolicy(BaseModel):
    max_attempts: int = Field(default=3, ge=1)
    retryable_categories: set[str] = Field(
        default_factory=lambda: {
            "timeout",
            "rate_limit_exceeded",
            "provider_unavailable",
            "transient",
            "connection_error",
        }
    )
    initial_backoff_seconds: float = Field(default=1.0, ge=0.0)
    backoff_factor: float = Field(default=2.0, ge=1.0)
    max_backoff_seconds: float = Field(default=30.0, ge=0.0)
    jitter: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "RetryPolicy":
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= initial_backoff_seconds")
        return self

    @staticmethod
    def _validate_attempt(attempt: int) -> None:
        if attempt < 1:
            raise ValueError("attempt must be >= 1")

    @staticmethod
    def _resolve_random_factor(
        rng: Callable[[], float] | float | None,
    ) -> float:
        if rng is None:
            value = random.uniform(-1.0, 1.0)
        elif callable(rng):
            value = float(rng())
        else:
            value = float(rng)

        if not -1.0 <= value <= 1.0:
            raise ValueError("jitter random factor must be between -1.0 and 1.0")

        return value

    def calculate_delay(
        self,
        attempt: int,
        rng: Callable[[], float] | float | None = None,
    ) -> float:
        self._validate_attempt(attempt)

        if self.initial_backoff_seconds == 0.0:
            return 0.0

        exponent = attempt - 1
        base_delay = self.initial_backoff_seconds * (self.backoff_factor**exponent)
        capped_delay = min(base_delay, self.max_backoff_seconds)

        if self.jitter == 0.0:
            return float(capped_delay)

        random_factor = self._resolve_random_factor(rng)
        jitter_offset = capped_delay * self.jitter * random_factor
        jittered_delay = max(0.0, float(capped_delay + jitter_offset))

        return min(jittered_delay, self.max_backoff_seconds)

    def evaluate(
        self,
        category: str,
        disposition: FailureDisposition | str,
        attempt: int,
        rng: Callable[[], float] | float | None = None,
    ) -> RetryDecision:
        self._validate_attempt(attempt)

        disp_str = disposition.value if isinstance(disposition, FailureDisposition) else disposition

        if attempt >= self.max_attempts:
            return RetryDecision(
                allowed=False,
                attempt=attempt,
                max_attempts=self.max_attempts,
                delay_seconds=0.0,
                reason=(f"Exceeded max attempts " f"({attempt}/{self.max_attempts})"),
            )

        if disp_str == FailureDisposition.AMBIGUOUS.value:
            return RetryDecision(
                allowed=False,
                attempt=attempt,
                max_attempts=self.max_attempts,
                delay_seconds=0.0,
                reason=("Ambiguous failure execution state " "cannot be safely retried"),
            )

        if disp_str == FailureDisposition.NON_RETRYABLE.value:
            return RetryDecision(
                allowed=False,
                attempt=attempt,
                max_attempts=self.max_attempts,
                delay_seconds=0.0,
                reason=(f"Failure disposition '{disp_str}' " "is non-retryable"),
            )

        if disp_str != FailureDisposition.RETRYABLE.value:
            return RetryDecision(
                allowed=False,
                attempt=attempt,
                max_attempts=self.max_attempts,
                delay_seconds=0.0,
                reason=f"Unknown failure disposition '{disp_str}'",
            )

        if category not in self.retryable_categories:
            return RetryDecision(
                allowed=False,
                attempt=attempt,
                max_attempts=self.max_attempts,
                delay_seconds=0.0,
                reason=(f"Category '{category}' is not configured " "as retryable in policy"),
            )

        delay = self.calculate_delay(attempt=attempt, rng=rng)

        return RetryDecision(
            allowed=True,
            attempt=attempt,
            max_attempts=self.max_attempts,
            delay_seconds=delay,
            reason=(
                f"Category '{category}' is retryable on " f"attempt {attempt}/{self.max_attempts}"
            ),
        )
