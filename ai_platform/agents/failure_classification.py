from __future__ import annotations

from enum import StrEnum

from ai_platform.agents.exceptions import AgentExecutionOwnershipLostError
from tools.models import ToolExecutionFailureCategory


class RuntimeFailureCategory(StrEnum):
    """Runtime-level semantic classification of an execution failure."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    AMBIGUOUS = "ambiguous"
    IN_PROGRESS = "in_progress"


def classify_runtime_failure(
    failure_category: ToolExecutionFailureCategory | None = None,
    *,
    retryable_failure_categories: frozenset[ToolExecutionFailureCategory] = frozenset(),
    exception: BaseException | None = None,
) -> RuntimeFailureCategory:
    """
    Translate an execution failure signal into runtime semantics.

    Tool execution policy remains authoritative for retryability.
    Ambiguous and in-progress outcomes have intrinsic runtime semantics
    and therefore take precedence over retry policy.
    Ownership loss is treated as an ambiguous execution outcome.
    """
    if isinstance(exception, AgentExecutionOwnershipLostError):
        return RuntimeFailureCategory.AMBIGUOUS

    if failure_category is ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS:
        return RuntimeFailureCategory.AMBIGUOUS

    if failure_category is ToolExecutionFailureCategory.EXECUTION_IN_PROGRESS:
        return RuntimeFailureCategory.IN_PROGRESS

    if failure_category is not None and failure_category in retryable_failure_categories:
        return RuntimeFailureCategory.RETRYABLE

    return RuntimeFailureCategory.NON_RETRYABLE
