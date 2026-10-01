import asyncio

from app.control_plane.retries.models import FailureClassification, FailureDisposition


class FailureClassifier:
    """Classify execution failures into durable retry dispositions."""

    NON_RETRYABLE_CATEGORIES: set[str] = {
        "approval_rejected",
        "authorization_denied",
        "authorization",
        "missing_principal",
        "validation_error",
        "schema_validation",
        "invalid_schema",
        "tenant_policy",
        "tool_not_found",
        "tool_disabled",
    }

    AMBIGUOUS_CATEGORIES: set[str] = {
        "ambiguous",
        "ambiguous_execution",
        "execution_ambiguous",
    }

    def classify(self, exc: Exception) -> FailureClassification:
        exc_type_name = type(exc).__name__
        exc_msg = str(exc)
        lower_msg = exc_msg.lower()

        if (
            exc_type_name in {"ApprovalRejectedError", "ApprovalRejected"}
            or "approval_rejected" in lower_msg
        ):
            return FailureClassification(
                category="approval_rejected",
                disposition=FailureDisposition.NON_RETRYABLE,
                reason=f"Approval rejected: {exc_msg}",
                details={"exception_type": exc_type_name},
            )

        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return FailureClassification(
                category="timeout",
                disposition=FailureDisposition.RETRYABLE,
                reason=f"Timeout occurred: {exc_msg or exc_type_name}",
                details={"exception_type": exc_type_name},
            )

        if isinstance(exc, ConnectionError):
            return FailureClassification(
                category="connection_error",
                disposition=FailureDisposition.RETRYABLE,
                reason=f"Connection failure: {exc_msg or exc_type_name}",
                details={"exception_type": exc_type_name},
            )

        if "rate_limit" in lower_msg or "rate limit" in lower_msg:
            return FailureClassification(
                category="rate_limit_exceeded",
                disposition=FailureDisposition.RETRYABLE,
                reason=f"Rate limit exceeded: {exc_msg}",
                details={"exception_type": exc_type_name},
            )

        if "provider_unavailable" in lower_msg or "service unavailable" in lower_msg:
            return FailureClassification(
                category="provider_unavailable",
                disposition=FailureDisposition.RETRYABLE,
                reason=f"Provider unavailable: {exc_msg}",
                details={"exception_type": exc_type_name},
            )

        if any(category in lower_msg for category in self.AMBIGUOUS_CATEGORIES):
            return FailureClassification(
                category="ambiguous_execution",
                disposition=FailureDisposition.AMBIGUOUS,
                reason=(f"Ambiguous execution outcome detected: {exc_msg}"),
                details={"exception_type": exc_type_name},
            )

        return FailureClassification(
            category="unknown",
            disposition=FailureDisposition.NON_RETRYABLE,
            reason=(f"Unclassified exception '{exc_type_name}': " f"{exc_msg}"),
            details={"exception_type": exc_type_name},
        )
