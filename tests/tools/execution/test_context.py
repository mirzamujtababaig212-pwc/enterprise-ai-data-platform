from __future__ import annotations

import pytest

from rag.governance import GovernancePolicy
from tools.execution.context import ToolExecutionContext


def test_tool_execution_context_defaults_to_empty_context() -> None:
    context = ToolExecutionContext()

    assert context.run_id is None
    assert context.call_id is None
    assert context.agent_name is None
    assert context.session_id is None
    assert context.user_id is None
    assert context.governance_policy is None
    assert context.request_metadata == {}


def test_tool_execution_context_preserves_execution_metadata() -> None:
    policy = GovernancePolicy(
        required_metadata={"classification": "internal"},
    )

    context = ToolExecutionContext(
        run_id="run-123",
        call_id="call-456",
        agent_name="research-agent",
        session_id="session-789",
        user_id="user-123",
        governance_policy=policy,
        request_metadata={"source": "api"},
    )

    assert context.run_id == "run-123"
    assert context.call_id == "call-456"
    assert context.agent_name == "research-agent"
    assert context.session_id == "session-789"
    assert context.user_id == "user-123"
    assert context.governance_policy is policy
    assert context.request_metadata == {"source": "api"}


def test_tool_execution_context_copies_request_metadata() -> None:
    metadata = {"source": "api"}

    context = ToolExecutionContext(
        request_metadata=metadata,
    )

    metadata["mutated"] = True

    assert context.request_metadata == {"source": "api"}


def test_tool_execution_context_is_immutable() -> None:
    context = ToolExecutionContext(
        run_id="run-123",
    )

    with pytest.raises(AttributeError):
        context.run_id = "run-456"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field",
    (
        "run_id",
        "call_id",
        "agent_name",
        "session_id",
        "user_id",
    ),
)
def test_tool_execution_context_rejects_empty_identity_fields(field: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ToolExecutionContext(**{field: "   "})


def test_tool_execution_context_rejects_invalid_governance_policy() -> None:
    with pytest.raises(
        TypeError,
        match="governance_policy must be a GovernancePolicy",
    ):
        ToolExecutionContext(
            governance_policy="internal-only",  # type: ignore[arg-type]
        )
