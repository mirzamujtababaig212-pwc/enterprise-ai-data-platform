from __future__ import annotations

import pytest

from ai_platform.agents.orchestration import OrchestrationStepStatus
from ai_platform.agents.plans import (
    ENTERPRISE_RAG_ANALYST_AGENT,
    build_agent_orchestration_plan,
    build_enterprise_rag_analyst_plan,
)


def test_enterprise_rag_analyst_plan_has_expected_steps() -> None:
    plan = build_enterprise_rag_analyst_plan()

    assert [step.step_id for step in plan.steps] == [
        "retrieve_evidence",
        "analyze_evidence",
        "produce_answer",
    ]

    assert [step.step_index for step in plan.steps] == [0, 1, 2]

    assert [step.name for step in plan.steps] == [
        "Retrieve enterprise evidence",
        "Analyze retrieved evidence",
        "Produce grounded answer",
    ]


def test_enterprise_rag_analyst_plan_starts_all_steps_pending() -> None:
    plan = build_enterprise_rag_analyst_plan()

    assert all(step.status is OrchestrationStepStatus.PENDING for step in plan.steps)


def test_enterprise_rag_analyst_plan_does_not_assign_tool_rounds() -> None:
    plan = build_enterprise_rag_analyst_plan()

    assert all(step.tool_round is None for step in plan.steps)


def test_enterprise_rag_analyst_plan_contains_phase_metadata() -> None:
    plan = build_enterprise_rag_analyst_plan()

    assert [step.metadata["phase"] for step in plan.steps] == [
        "evidence_retrieval",
        "analysis",
        "response_generation",
    ]


def test_enterprise_rag_analyst_plan_materializes_independent_state() -> None:
    first_plan = build_enterprise_rag_analyst_plan()
    second_plan = build_enterprise_rag_analyst_plan()

    first_state = first_plan.materialize_state()
    second_state = second_plan.materialize_state()

    first_state.start_step(0)

    assert first_state.current_step_index == 0
    assert second_state.current_step_index is None

    assert all(step.status is OrchestrationStepStatus.PENDING for step in second_plan.steps)


def test_build_agent_orchestration_plan_resolves_known_agent() -> None:
    plan = build_agent_orchestration_plan(ENTERPRISE_RAG_ANALYST_AGENT)

    assert [step.step_id for step in plan.steps] == [
        "retrieve_evidence",
        "analyze_evidence",
        "produce_answer",
    ]


def test_build_agent_orchestration_plan_rejects_unknown_agent() -> None:
    with pytest.raises(LookupError, match="No orchestration plan"):
        build_agent_orchestration_plan("unknown-agent")


@pytest.mark.parametrize("agent_name", ["", "   ", None])
def test_build_agent_orchestration_plan_rejects_invalid_agent_name(
    agent_name,
) -> None:
    with pytest.raises(ValueError, match="Agent name"):
        build_agent_orchestration_plan(agent_name)
