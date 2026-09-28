from ai_platform.agents.exceptions import AgentExecutionOwnershipLostError
from ai_platform.agents.failure_classification import (
    RuntimeFailureCategory,
    classify_runtime_failure,
)
from tools.models import ToolExecutionFailureCategory


def test_classifies_ambiguous_tool_execution_as_ambiguous() -> None:
    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS,
        )
        is RuntimeFailureCategory.AMBIGUOUS
    )


def test_classifies_in_progress_tool_execution_as_in_progress() -> None:
    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.EXECUTION_IN_PROGRESS,
        )
        is RuntimeFailureCategory.IN_PROGRESS
    )


def test_classifies_policy_selected_retryable_failure_as_retryable() -> None:
    retryable = frozenset(
        {
            ToolExecutionFailureCategory.EXECUTION_ERROR,
            ToolExecutionFailureCategory.TIMEOUT,
        }
    )

    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.EXECUTION_ERROR,
            retryable_failure_categories=retryable,
        )
        is RuntimeFailureCategory.RETRYABLE
    )

    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.TIMEOUT,
            retryable_failure_categories=retryable,
        )
        is RuntimeFailureCategory.RETRYABLE
    )


def test_classifies_policy_excluded_failure_as_non_retryable() -> None:
    retryable = frozenset(
        {
            ToolExecutionFailureCategory.EXECUTION_ERROR,
        }
    )

    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.AUTHORIZATION,
            retryable_failure_categories=retryable,
        )
        is RuntimeFailureCategory.NON_RETRYABLE
    )


def test_retryability_is_policy_driven() -> None:
    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.EXECUTION_ERROR,
        )
        is RuntimeFailureCategory.NON_RETRYABLE
    )

    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.EXECUTION_ERROR,
            retryable_failure_categories=frozenset({ToolExecutionFailureCategory.EXECUTION_ERROR}),
        )
        is RuntimeFailureCategory.RETRYABLE
    )


def test_ownership_loss_is_ambiguous() -> None:
    assert (
        classify_runtime_failure(
            exception=AgentExecutionOwnershipLostError("durable run ownership was lost"),
        )
        is RuntimeFailureCategory.AMBIGUOUS
    )


def test_ambiguous_and_in_progress_override_retry_policy() -> None:
    retryable = frozenset(
        {
            ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS,
            ToolExecutionFailureCategory.EXECUTION_IN_PROGRESS,
        }
    )

    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS,
            retryable_failure_categories=retryable,
        )
        is RuntimeFailureCategory.AMBIGUOUS
    )

    assert (
        classify_runtime_failure(
            ToolExecutionFailureCategory.EXECUTION_IN_PROGRESS,
            retryable_failure_categories=retryable,
        )
        is RuntimeFailureCategory.IN_PROGRESS
    )


def test_missing_failure_signal_is_non_retryable() -> None:
    assert classify_runtime_failure() is RuntimeFailureCategory.NON_RETRYABLE
