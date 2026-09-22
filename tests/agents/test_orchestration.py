import pytest

from ai_platform.agents.orchestration import (
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
