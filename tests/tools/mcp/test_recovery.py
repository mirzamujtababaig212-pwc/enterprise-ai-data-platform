from __future__ import annotations

import pytest

from tools.mcp.recovery import MCPRecoveryPolicy


def test_recovery_policy_defaults():
    policy = MCPRecoveryPolicy()

    assert policy.max_attempts == 3
    assert policy.initial_backoff == 1.0
    assert policy.max_backoff == 30.0
    assert policy.cooldown == 60.0


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [
        (1, 1.0),
        (2, 2.0),
        (3, 4.0),
        (4, 8.0),
    ],
)
def test_recovery_policy_calculates_exponential_backoff(
    attempt: int,
    expected: float,
):
    policy = MCPRecoveryPolicy(
        initial_backoff=1.0,
        max_backoff=30.0,
    )

    assert policy.backoff_for_attempt(attempt) == expected


def test_recovery_policy_caps_backoff():
    policy = MCPRecoveryPolicy(
        initial_backoff=5.0,
        max_backoff=10.0,
    )

    assert policy.backoff_for_attempt(1) == 5.0
    assert policy.backoff_for_attempt(2) == 10.0
    assert policy.backoff_for_attempt(3) == 10.0
    assert policy.backoff_for_attempt(10) == 10.0


@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_recovery_policy_allows_attempts_within_budget(attempt: int):
    policy = MCPRecoveryPolicy(max_attempts=3)

    assert policy.allows_attempt(attempt) is True


@pytest.mark.parametrize("attempt", [4, 5, 10])
def test_recovery_policy_rejects_attempts_after_budget(attempt: int):
    policy = MCPRecoveryPolicy(max_attempts=3)

    assert policy.allows_attempt(attempt) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"initial_backoff": -1.0},
        {"max_backoff": -1.0},
        {"cooldown": -1.0},
        {"initial_backoff": 10.0, "max_backoff": 5.0},
    ],
)
def test_recovery_policy_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        MCPRecoveryPolicy(**kwargs)


@pytest.mark.parametrize("attempt", [0, -1])
def test_recovery_policy_rejects_invalid_attempt(attempt: int):
    policy = MCPRecoveryPolicy()

    with pytest.raises(
        ValueError,
        match="attempt must be at least one",
    ):
        policy.backoff_for_attempt(attempt)

    with pytest.raises(
        ValueError,
        match="attempt must be at least one",
    ):
        policy.allows_attempt(attempt)
