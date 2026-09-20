import pytest

from tools.models import (
    ToolDefinition,
    ToolExecutionFailureCategory,
    ToolExecutionPolicy,
)


def test_tool_execution_policy_defaults_to_no_retries():
    policy = ToolExecutionPolicy()

    assert policy.max_retries == 0
    assert policy.retryable_failure_categories == frozenset()
    assert policy.backoff_seconds == 0.0


def test_tool_execution_policy_accepts_explicit_retry_configuration():
    policy = ToolExecutionPolicy(
        max_retries=3,
        retryable_failure_categories=frozenset(
            {
                ToolExecutionFailureCategory.TIMEOUT,
                ToolExecutionFailureCategory.EXECUTION_ERROR,
            }
        ),
        backoff_seconds=0.25,
    )

    assert policy.max_retries == 3
    assert policy.retryable_failure_categories == frozenset(
        {
            ToolExecutionFailureCategory.TIMEOUT,
            ToolExecutionFailureCategory.EXECUTION_ERROR,
        }
    )
    assert policy.backoff_seconds == 0.25


@pytest.mark.parametrize("value", [-1, -10])
def test_tool_execution_policy_rejects_negative_retries(value):
    with pytest.raises(
        ValueError,
        match="max_retries must be greater than or equal to zero",
    ):
        ToolExecutionPolicy(max_retries=value)


def test_tool_execution_policy_rejects_boolean_retry_count():
    with pytest.raises(TypeError, match="max_retries must be an integer"):
        ToolExecutionPolicy(max_retries=True)


def test_tool_execution_policy_rejects_negative_backoff():
    with pytest.raises(
        ValueError,
        match="backoff_seconds must be greater than or equal to zero",
    ):
        ToolExecutionPolicy(backoff_seconds=-0.1)


def test_tool_execution_policy_rejects_invalid_failure_category():
    with pytest.raises(
        TypeError,
        match="retryable_failure_categories must contain",
    ):
        ToolExecutionPolicy(
            retryable_failure_categories=frozenset({"timeout"}),
        )


def test_tool_definition_defaults_to_no_retry_policy():
    definition = ToolDefinition(
        name="test_tool",
        description="A test tool.",
    )

    assert definition.execution_policy == ToolExecutionPolicy()


def test_execution_in_progress_is_not_retryable_by_default() -> None:
    policy = ToolExecutionPolicy()

    assert (
        ToolExecutionFailureCategory.EXECUTION_IN_PROGRESS
        not in policy.retryable_failure_categories
    )
