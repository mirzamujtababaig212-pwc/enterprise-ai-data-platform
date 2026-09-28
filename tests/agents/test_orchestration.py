import pytest

from ai_platform.agents.orchestration import (
    AgentRuntimeDecision,
    AgentRuntimePhase,
    AgentRuntimeState,
    OrchestrationStepResult,
    OrchestrationPlan,
    OrchestrationState,
    OrchestrationStep,
    OrchestrationStepCompletionPolicy,
    OrchestrationStepStatus,
)


def test_orchestration_step_defaults_to_agent_response_completion() -> None:
    step = OrchestrationStep(
        step_id="step-1",
        step_index=0,
        name="Retrieve evidence",
        status=OrchestrationStepStatus.PENDING,
    )

    assert step.completion_policy is OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE


def test_orchestration_step_rejects_invalid_completion_policy() -> None:
    with pytest.raises(TypeError, match="completion_policy"):
        OrchestrationStep(
            step_id="step-1",
            step_index=0,
            name="Retrieve evidence",
            status=OrchestrationStepStatus.PENDING,
            completion_policy="on_agent_response",
        )


def test_orchestration_step_accepts_valid_state() -> None:
    step = OrchestrationStep(
        step_id="step-1",
        step_index=0,
        name="Retrieve evidence",
        status=OrchestrationStepStatus.RUNNING,
        tool_round=1,
        metadata={"source": "agent"},
    )

    assert step.step_id == "step-1"
    assert step.step_index == 0
    assert step.name == "Retrieve evidence"
    assert step.status is OrchestrationStepStatus.RUNNING
    assert step.tool_round == 1
    assert step.metadata == {"source": "agent"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_id": "", "step_index": 0, "name": "step"},
        {"step_id": "step", "step_index": -1, "name": "step"},
        {"step_id": "step", "step_index": 0, "name": ""},
    ],
)
def test_orchestration_step_rejects_invalid_identity(kwargs) -> None:
    with pytest.raises(ValueError):
        OrchestrationStep(
            **kwargs,
            status=OrchestrationStepStatus.PENDING,
        )


def test_orchestration_step_rejects_negative_tool_round() -> None:
    with pytest.raises(ValueError, match="tool_round"):
        OrchestrationStep(
            step_id="step-1",
            step_index=0,
            name="Retrieve",
            status=OrchestrationStepStatus.RUNNING,
            tool_round=-1,
        )


def test_orchestration_step_copies_metadata() -> None:
    metadata = {"source": "agent"}

    step = OrchestrationStep(
        step_id="step-1",
        step_index=0,
        name="Retrieve",
        status=OrchestrationStepStatus.COMPLETED,
        metadata=metadata,
    )

    metadata["source"] = "mutated"

    assert step.metadata == {"source": "agent"}


def test_orchestration_state_accepts_steps() -> None:
    step = OrchestrationStep(
        step_id="step-1",
        step_index=0,
        name="Retrieve",
        status=OrchestrationStepStatus.COMPLETED,
    )

    state = OrchestrationState(
        steps=[step],
        current_step_index=0,
    )

    assert state.steps == [step]
    assert state.current_step_index == 0


def test_orchestration_state_copies_steps() -> None:
    steps = [
        OrchestrationStep(
            step_id="step-1",
            step_index=0,
            name="Retrieve",
            status=OrchestrationStepStatus.COMPLETED,
        )
    ]

    state = OrchestrationState(steps=steps)

    steps.clear()

    assert len(state.steps) == 1


def test_orchestration_state_rejects_invalid_step() -> None:
    with pytest.raises(TypeError, match="Orchestration steps"):
        OrchestrationState(steps=["invalid"])


def test_orchestration_state_rejects_negative_current_index() -> None:
    with pytest.raises(ValueError, match="current_step_index"):
        OrchestrationState(current_step_index=-1)


def test_orchestration_state_current_step() -> None:
    steps = [
        OrchestrationStep(
            step_id="retrieve",
            step_index=0,
            name="Retrieve evidence",
            status=OrchestrationStepStatus.PENDING,
        ),
        OrchestrationStep(
            step_id="answer",
            step_index=1,
            name="Produce answer",
            status=OrchestrationStepStatus.PENDING,
        ),
    ]

    state = OrchestrationState(steps=steps)

    assert state.current_step is None

    current = state.start_step(0)

    assert current.status is OrchestrationStepStatus.RUNNING
    assert state.current_step is current
    assert state.current_step_index == 0


def test_orchestration_state_completes_and_advances() -> None:
    steps = [
        OrchestrationStep(
            step_id="retrieve",
            step_index=0,
            name="Retrieve evidence",
            status=OrchestrationStepStatus.PENDING,
        ),
        OrchestrationStep(
            step_id="answer",
            step_index=1,
            name="Produce answer",
            status=OrchestrationStepStatus.PENDING,
        ),
    ]

    state = OrchestrationState(steps=steps)

    state.start_step(0)
    completed = state.complete_step(tool_round=1)

    assert completed.status is OrchestrationStepStatus.COMPLETED
    assert completed.tool_round == 1

    next_step = state.advance()

    assert next_step is not None
    assert next_step.step_id == "answer"
    assert next_step.status is OrchestrationStepStatus.RUNNING
    assert state.current_step_index == 1


def test_orchestration_state_cannot_advance_without_current_step() -> None:
    state = OrchestrationState(
        steps=[
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve evidence",
                status=OrchestrationStepStatus.PENDING,
            )
        ]
    )

    with pytest.raises(ValueError, match="without a current step"):
        state.advance()


def test_orchestration_state_cannot_advance_failed_step() -> None:
    state = OrchestrationState(
        steps=[
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve evidence",
                status=OrchestrationStepStatus.PENDING,
            ),
            OrchestrationStep(
                step_id="answer",
                step_index=1,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
            ),
        ]
    )

    state.start_step(0)
    state.fail_step(metadata={"reason": "timeout"})

    with pytest.raises(ValueError, match="COMPLETED"):
        state.advance()


def test_orchestration_state_advance_preserves_completed_step() -> None:
    state = OrchestrationState(
        steps=[
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve evidence",
                status=OrchestrationStepStatus.PENDING,
            ),
            OrchestrationStep(
                step_id="analyze",
                step_index=1,
                name="Analyze evidence",
                status=OrchestrationStepStatus.PENDING,
            ),
        ]
    )

    state.start_step(0)
    completed = state.complete_step(tool_round=2)

    next_step = state.advance()

    assert completed.status is OrchestrationStepStatus.COMPLETED
    assert completed.tool_round == 2

    assert state.steps[0] is completed
    assert state.steps[0].status is OrchestrationStepStatus.COMPLETED
    assert state.steps[0].tool_round == 2

    assert next_step is not None
    assert next_step.status is OrchestrationStepStatus.RUNNING
    assert next_step.tool_round is None


def test_orchestration_state_finishes_after_last_step() -> None:
    step = OrchestrationStep(
        step_id="answer",
        step_index=0,
        name="Produce answer",
        status=OrchestrationStepStatus.PENDING,
    )

    state = OrchestrationState(steps=[step])

    state.start_step(0)
    state.complete_step()

    assert state.advance() is None
    assert state.current_step_index is None


def test_orchestration_state_failure_preserves_failure_metadata() -> None:
    step = OrchestrationStep(
        step_id="retrieve",
        step_index=0,
        name="Retrieve evidence",
        status=OrchestrationStepStatus.PENDING,
        metadata={"source": "vehicle"},
    )

    state = OrchestrationState(steps=[step])

    state.start_step(0)
    failed = state.fail_step(metadata={"reason": "timeout"})

    assert failed.status is OrchestrationStepStatus.FAILED
    assert failed.metadata == {
        "source": "vehicle",
        "reason": "timeout",
    }


def test_orchestration_state_rejects_out_of_range_current_step() -> None:
    with pytest.raises(ValueError, match="existing step"):
        OrchestrationState(current_step_index=0)


def test_orchestration_state_requires_completion_before_advance() -> None:
    step = OrchestrationStep(
        step_id="retrieve",
        step_index=0,
        name="Retrieve evidence",
        status=OrchestrationStepStatus.PENDING,
    )

    state = OrchestrationState(steps=[step])
    state.start_step(0)

    with pytest.raises(ValueError, match="COMPLETED"):
        state.advance()


def test_orchestration_state_rejects_invalid_transition() -> None:
    step = OrchestrationStep(
        step_id="retrieve",
        step_index=0,
        name="Retrieve evidence",
        status=OrchestrationStepStatus.COMPLETED,
    )

    state = OrchestrationState(steps=[step])

    with pytest.raises(ValueError, match="cannot start"):
        state.start_step(0)


def test_orchestration_plan_accepts_valid_pending_steps() -> None:
    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve evidence",
                status=OrchestrationStepStatus.PENDING,
            ),
            OrchestrationStep(
                step_id="answer",
                step_index=1,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
            ),
        )
    )

    assert [step.step_id for step in plan.steps] == ["retrieve", "answer"]


def test_orchestration_plan_rejects_empty_steps() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        OrchestrationPlan()


def test_orchestration_plan_rejects_non_step() -> None:
    with pytest.raises(TypeError, match="Orchestration plan steps"):
        OrchestrationPlan(steps=("invalid",))


def test_orchestration_plan_rejects_duplicate_step_ids() -> None:
    with pytest.raises(ValueError, match="duplicate step_id"):
        OrchestrationPlan(
            steps=(
                OrchestrationStep(
                    step_id="retrieve",
                    step_index=0,
                    name="Retrieve",
                    status=OrchestrationStepStatus.PENDING,
                ),
                OrchestrationStep(
                    step_id="retrieve",
                    step_index=1,
                    name="Retrieve again",
                    status=OrchestrationStepStatus.PENDING,
                ),
            )
        )


@pytest.mark.parametrize(
    "steps",
    [
        (
            OrchestrationStep(
                step_id="retrieve",
                step_index=1,
                name="Retrieve",
                status=OrchestrationStepStatus.PENDING,
            ),
        ),
        (
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve",
                status=OrchestrationStepStatus.PENDING,
            ),
            OrchestrationStep(
                step_id="answer",
                step_index=2,
                name="Answer",
                status=OrchestrationStepStatus.PENDING,
            ),
        ),
    ],
)
def test_orchestration_plan_rejects_non_contiguous_indexes(steps) -> None:
    with pytest.raises(ValueError, match="contiguous"):
        OrchestrationPlan(steps=steps)


@pytest.mark.parametrize(
    "status",
    [
        OrchestrationStepStatus.RUNNING,
        OrchestrationStepStatus.COMPLETED,
        OrchestrationStepStatus.FAILED,
    ],
)
def test_orchestration_plan_requires_pending_steps(
    status: OrchestrationStepStatus,
) -> None:
    with pytest.raises(ValueError, match="PENDING"):
        OrchestrationPlan(
            steps=(
                OrchestrationStep(
                    step_id="retrieve",
                    step_index=0,
                    name="Retrieve",
                    status=status,
                ),
            )
        )


def test_orchestration_plan_rejects_runtime_tool_round() -> None:
    with pytest.raises(ValueError, match="runtime tool_round"):
        OrchestrationPlan(
            steps=(
                OrchestrationStep(
                    step_id="retrieve",
                    step_index=0,
                    name="Retrieve",
                    status=OrchestrationStepStatus.PENDING,
                    tool_round=1,
                ),
            )
        )


def test_orchestration_plan_materializes_fresh_state() -> None:
    plan = OrchestrationPlan(
        steps=(
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve evidence",
                status=OrchestrationStepStatus.PENDING,
                metadata={"source": "vehicle"},
            ),
            OrchestrationStep(
                step_id="answer",
                step_index=1,
                name="Produce answer",
                status=OrchestrationStepStatus.PENDING,
            ),
        )
    )

    first = plan.materialize_state()
    second = plan.materialize_state()

    assert first is not second
    assert first.steps == second.steps
    assert first.current_step is None
    assert second.current_step is None

    first.start_step(0)

    assert first.current_step_index == 0
    assert second.current_step_index is None
    assert plan.steps[0].status is OrchestrationStepStatus.PENDING


def test_orchestration_state_stores_and_retrieves_step_result() -> None:
    steps = [
        OrchestrationStep(
            step_id="retrieve",
            step_index=0,
            name="Retrieve",
            status=OrchestrationStepStatus.PENDING,
        ),
        OrchestrationStep(
            step_id="answer",
            step_index=1,
            name="Answer",
            status=OrchestrationStepStatus.PENDING,
        ),
    ]
    state = OrchestrationState(steps=steps)

    result = state.set_step_result(
        "retrieve",
        {"documents": ["doc-1", "doc-2"]},
        metadata={"count": 2},
    )

    assert result.step_id == "retrieve"
    assert result.output == {"documents": ["doc-1", "doc-2"]}
    assert result.metadata == {"count": 2}
    assert state.get_step_result("retrieve") is result
    assert state.get_step_result("answer") is None


def test_orchestration_state_rejects_consuming_running_step_result() -> None:
    state = OrchestrationState(
        steps=[
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve",
                status=OrchestrationStepStatus.PENDING,
            )
        ]
    )

    state.start_step(0)
    state.set_step_result("retrieve", {"documents": ["doc-1"]})

    with pytest.raises(ValueError, match="must be COMPLETED"):
        state.get_completed_step_result("retrieve")


def test_orchestration_state_rejects_completed_step_without_result() -> None:
    state = OrchestrationState(
        steps=[
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve",
                status=OrchestrationStepStatus.PENDING,
            )
        ]
    )

    state.start_step(0)
    state.complete_step()

    with pytest.raises(ValueError, match="COMPLETED but has no stored result"):
        state.get_completed_step_result("retrieve")


def test_orchestration_state_returns_completed_step_result() -> None:
    state = OrchestrationState(
        steps=[
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve",
                status=OrchestrationStepStatus.PENDING,
            )
        ]
    )

    state.start_step(0)

    result = state.set_step_result(
        "retrieve",
        {"documents": ["doc-1", "doc-2"]},
        metadata={"retrieved_count": 2},
    )

    state.complete_step()

    completed_result = state.get_completed_step_result("retrieve")

    assert completed_result is result
    assert completed_result.output == {"documents": ["doc-1", "doc-2"]}
    assert completed_result.metadata == {"retrieved_count": 2}


def test_orchestration_state_step_result_metadata_is_copied() -> None:
    steps = [
        OrchestrationStep(
            step_id="retrieve",
            step_index=0,
            name="Retrieve",
            status=OrchestrationStepStatus.PENDING,
        )
    ]
    state = OrchestrationState(steps=steps)
    metadata = {"source": "rag"}

    result = state.set_step_result(
        "retrieve",
        "evidence",
        metadata=metadata,
    )

    metadata["source"] = "mutated"

    assert result.metadata == {"source": "rag"}


def test_orchestration_state_rejects_unknown_step_result() -> None:
    state = OrchestrationState(
        steps=[
            OrchestrationStep(
                step_id="retrieve",
                step_index=0,
                name="Retrieve",
                status=OrchestrationStepStatus.PENDING,
            )
        ]
    )

    with pytest.raises(ValueError, match="Unknown orchestration step 'missing'"):
        state.set_step_result("missing", "output")


def test_orchestration_state_rejects_result_key_mismatch() -> None:
    step = OrchestrationStep(
        step_id="retrieve",
        step_index=0,
        name="Retrieve",
        status=OrchestrationStepStatus.PENDING,
    )

    result = OrchestrationStepResult(
        step_id="different",
        output="output",
    )

    with pytest.raises(ValueError, match="result key must match"):
        OrchestrationState(
            steps=[step],
            step_results={"retrieve": result},
        )


def test_orchestration_state_results_are_independent_per_state() -> None:
    steps = [
        OrchestrationStep(
            step_id="retrieve",
            step_index=0,
            name="Retrieve",
            status=OrchestrationStepStatus.PENDING,
        )
    ]

    first = OrchestrationState(steps=steps)
    second = OrchestrationState(steps=steps)

    first.set_step_result("retrieve", "first")

    assert first.get_step_result("retrieve") is not None
    assert second.get_step_result("retrieve") is None


def test_runtime_phase_contract_allows_valid_loop_transitions():
    from ai_platform.agents.orchestration import (
        AgentRuntimePhase,
        validate_runtime_phase_transition,
    )

    transitions = [
        (AgentRuntimePhase.PLAN, AgentRuntimePhase.ACT),
        (AgentRuntimePhase.ACT, AgentRuntimePhase.OBSERVE),
        (AgentRuntimePhase.OBSERVE, AgentRuntimePhase.EVALUATE),
    ]

    for current, target in transitions:
        validate_runtime_phase_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        ("plan", "plan"),
        ("plan", "observe"),
        ("plan", "evaluate"),
        ("act", "plan"),
        ("act", "evaluate"),
        ("observe", "plan"),
        ("observe", "act"),
        ("evaluate", "plan"),
        ("evaluate", "act"),
    ],
)
def test_runtime_phase_contract_rejects_invalid_transitions(current, target):
    from ai_platform.agents.orchestration import (
        AgentRuntimePhase,
        validate_runtime_phase_transition,
    )

    with pytest.raises(ValueError, match="Invalid agent runtime phase transition"):
        validate_runtime_phase_transition(
            AgentRuntimePhase(current),
            AgentRuntimePhase(target),
        )


@pytest.mark.parametrize(
    "decision",
    [
        "continue",
        "stop",
    ],
)
def test_runtime_decision_is_valid_only_from_evaluate(decision):
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        validate_runtime_decision,
    )

    validate_runtime_decision(
        AgentRuntimePhase.EVALUATE,
        AgentRuntimeDecision(decision),
    )


@pytest.mark.parametrize(
    "phase",
    [
        "plan",
        "act",
        "observe",
    ],
)
def test_runtime_decision_is_rejected_before_evaluate(phase):
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        validate_runtime_decision,
    )

    with pytest.raises(ValueError, match="only valid from the evaluate phase"):
        validate_runtime_decision(
            AgentRuntimePhase(phase),
            AgentRuntimeDecision.CONTINUE,
        )


def test_runtime_state_starts_at_plan_without_decision() -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    state = AgentRuntimeState()

    assert state.phase is AgentRuntimePhase.PLAN
    assert state.decision is None
    assert state.current_step_index is None
    assert state.iteration == 1
    assert AgentRuntimeDecision.CONTINUE.value == "continue"


def test_runtime_state_progresses_through_semantic_loop() -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    state = AgentRuntimeState()

    state.transition_to(
        AgentRuntimePhase.ACT,
        current_step_index=0,
    )
    assert state.phase is AgentRuntimePhase.ACT
    assert state.current_step_index == 0
    assert state.decision is None

    state.transition_to(AgentRuntimePhase.OBSERVE)
    assert state.phase is AgentRuntimePhase.OBSERVE
    assert state.current_step_index is None

    state.transition_to(AgentRuntimePhase.EVALUATE)
    assert state.phase is AgentRuntimePhase.EVALUATE
    assert state.decision is None

    state.evaluate(AgentRuntimeDecision.CONTINUE)

    assert state.decision is AgentRuntimeDecision.CONTINUE


def test_runtime_state_continue_returns_to_plan() -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    state = AgentRuntimeState()

    state.transition_to(AgentRuntimePhase.ACT)
    state.transition_to(AgentRuntimePhase.OBSERVE)
    state.transition_to(AgentRuntimePhase.EVALUATE)
    state.evaluate(AgentRuntimeDecision.CONTINUE)

    state.continue_to_plan(current_step_index=1)

    assert state.phase is AgentRuntimePhase.PLAN
    assert state.decision is None
    assert state.current_step_index == 1
    assert state.iteration == 2


def test_runtime_state_stop_is_terminal_decision() -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    state = AgentRuntimeState()

    state.transition_to(AgentRuntimePhase.ACT)
    state.transition_to(AgentRuntimePhase.OBSERVE)
    state.transition_to(AgentRuntimePhase.EVALUATE)
    state.evaluate(AgentRuntimeDecision.STOP)

    state.stop()

    assert state.phase is AgentRuntimePhase.EVALUATE
    assert state.decision is AgentRuntimeDecision.STOP


@pytest.mark.parametrize(
    "phase",
    [
        "plan",
        "act",
        "observe",
    ],
)
def test_runtime_state_rejects_decision_before_evaluate(phase) -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    state = AgentRuntimeState(phase=AgentRuntimePhase(phase))

    with pytest.raises(ValueError, match="only valid from the evaluate phase"):
        state.evaluate(AgentRuntimeDecision.CONTINUE)


def test_runtime_state_rejects_continue_without_continue_decision() -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    state = AgentRuntimeState()
    state.transition_to(AgentRuntimePhase.ACT)
    state.transition_to(AgentRuntimePhase.OBSERVE)
    state.transition_to(AgentRuntimePhase.EVALUATE)

    with pytest.raises(ValueError, match="CONTINUE decision"):
        state.continue_to_plan()

    state.evaluate(AgentRuntimeDecision.STOP)

    with pytest.raises(ValueError, match="CONTINUE decision"):
        state.continue_to_plan()


def test_runtime_state_rejects_stop_without_stop_decision() -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimeDecision,
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    state = AgentRuntimeState()
    state.transition_to(AgentRuntimePhase.ACT)
    state.transition_to(AgentRuntimePhase.OBSERVE)
    state.transition_to(AgentRuntimePhase.EVALUATE)

    with pytest.raises(ValueError, match="STOP decision"):
        state.stop()

    state.evaluate(AgentRuntimeDecision.CONTINUE)

    with pytest.raises(ValueError, match="STOP decision"):
        state.stop()


@pytest.mark.parametrize(
    ("phase", "target"),
    [
        ("plan", "observe"),
        ("act", "evaluate"),
        ("observe", "plan"),
        ("evaluate", "act"),
    ],
)
def test_runtime_state_rejects_invalid_phase_transitions(phase, target) -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    state = AgentRuntimeState(phase=AgentRuntimePhase(phase))

    with pytest.raises(ValueError, match="Invalid agent runtime phase transition"):
        state.transition_to(AgentRuntimePhase(target))


def test_runtime_state_rejects_invalid_current_step_index() -> None:
    from ai_platform.agents.orchestration import (
        AgentRuntimePhase,
        AgentRuntimeState,
    )

    with pytest.raises(
        ValueError,
        match="current_step_index must not be negative",
    ):
        AgentRuntimeState(current_step_index=-1)

    state = AgentRuntimeState()

    with pytest.raises(
        ValueError,
        match="current_step_index must not be negative",
    ):
        state.transition_to(
            AgentRuntimePhase.ACT,
            current_step_index=-1,
        )


def test_runtime_state_to_metadata_serializes_plan_state():
    state = AgentRuntimeState(current_step_index=0)

    assert state.to_metadata() == {
        "runtime": {
            "phase": "plan",
            "decision": None,
            "current_step_index": 0,
            "iteration": 1,
        }
    }


def test_runtime_state_to_metadata_serializes_evaluate_continue_state():
    state = AgentRuntimeState(
        phase=AgentRuntimePhase.EVALUATE,
        current_step_index=2,
    )
    state.evaluate(AgentRuntimeDecision.CONTINUE)

    assert state.to_metadata() == {
        "runtime": {
            "phase": "evaluate",
            "decision": "continue",
            "current_step_index": 2,
            "iteration": 1,
        }
    }


def test_runtime_state_to_metadata_serializes_evaluate_stop_state():
    state = AgentRuntimeState(
        phase=AgentRuntimePhase.EVALUATE,
        current_step_index=3,
    )
    state.evaluate(AgentRuntimeDecision.STOP)
    state.stop()

    assert state.to_metadata() == {
        "runtime": {
            "phase": "evaluate",
            "decision": "stop",
            "current_step_index": 3,
            "iteration": 1,
        }
    }


@pytest.mark.parametrize(
    "state",
    [
        AgentRuntimeState(
            phase=AgentRuntimePhase.PLAN,
            current_step_index=0,
        ),
        AgentRuntimeState(
            phase=AgentRuntimePhase.ACT,
            current_step_index=1,
        ),
        AgentRuntimeState(
            phase=AgentRuntimePhase.OBSERVE,
            current_step_index=1,
        ),
    ],
)
def test_runtime_state_metadata_round_trip_without_decision(state):
    restored = AgentRuntimeState.from_metadata(state.to_metadata())

    assert restored == state


@pytest.mark.parametrize(
    "decision",
    [
        AgentRuntimeDecision.CONTINUE,
        AgentRuntimeDecision.STOP,
    ],
)
def test_runtime_state_metadata_round_trip_from_evaluate(decision):
    state = AgentRuntimeState(
        phase=AgentRuntimePhase.EVALUATE,
        current_step_index=2,
    )
    state.evaluate(decision)

    restored = AgentRuntimeState.from_metadata(state.to_metadata())

    assert restored == state


def test_runtime_state_from_metadata_returns_none_when_runtime_missing():
    assert AgentRuntimeState.from_metadata({"source": "test"}) is None


def test_runtime_state_from_metadata_rejects_non_dictionary_runtime():
    with pytest.raises(
        ValueError,
        match="dictionary under 'runtime'",
    ):
        AgentRuntimeState.from_metadata({"runtime": "invalid"})


def test_runtime_state_from_metadata_rejects_missing_phase():
    with pytest.raises(
        ValueError,
        match="missing 'phase'",
    ):
        AgentRuntimeState.from_metadata({"runtime": {}})


def test_runtime_state_from_metadata_rejects_invalid_phase():
    with pytest.raises(
        ValueError,
        match="invalid phase",
    ):
        AgentRuntimeState.from_metadata(
            {
                "runtime": {
                    "phase": "invalid",
                    "decision": None,
                    "current_step_index": 0,
                }
            }
        )


def test_runtime_state_from_metadata_rejects_invalid_decision():
    with pytest.raises(
        ValueError,
        match="invalid decision",
    ):
        AgentRuntimeState.from_metadata(
            {
                "runtime": {
                    "phase": "evaluate",
                    "decision": "invalid",
                    "current_step_index": 0,
                }
            }
        )


def test_runtime_state_from_metadata_rejects_decision_before_evaluate():
    with pytest.raises(
        ValueError,
        match="decision must be None before the evaluate phase",
    ):
        AgentRuntimeState.from_metadata(
            {
                "runtime": {
                    "phase": "act",
                    "decision": "continue",
                    "current_step_index": 0,
                }
            }
        )


def test_runtime_state_continue_increments_iteration() -> None:
    state = AgentRuntimeState()

    state.transition_to(AgentRuntimePhase.ACT)
    state.transition_to(AgentRuntimePhase.OBSERVE)
    state.transition_to(AgentRuntimePhase.EVALUATE)
    state.evaluate(AgentRuntimeDecision.CONTINUE)

    state.continue_to_plan()

    assert state.phase is AgentRuntimePhase.PLAN
    assert state.iteration == 2

    state.transition_to(AgentRuntimePhase.ACT)
    state.transition_to(AgentRuntimePhase.OBSERVE)
    state.transition_to(AgentRuntimePhase.EVALUATE)
    state.evaluate(AgentRuntimeDecision.CONTINUE)

    state.continue_to_plan()

    assert state.iteration == 3


@pytest.mark.parametrize("iteration", [0, -1])
def test_runtime_state_rejects_non_positive_iteration(iteration: int) -> None:
    with pytest.raises(
        ValueError,
        match="iteration must be greater than zero",
    ):
        AgentRuntimeState(iteration=iteration)


def test_runtime_state_metadata_defaults_legacy_iteration() -> None:
    restored = AgentRuntimeState.from_metadata(
        {
            "runtime": {
                "phase": "plan",
                "decision": None,
                "current_step_index": 0,
            }
        }
    )

    assert restored is not None
    assert restored.iteration == 1
