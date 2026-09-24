from __future__ import annotations

import asyncio

import pytest

from ai_platform.agents.lifecycle import (
    AgentExecutionLifecyclePhase,
    AgentExecutionLifecycleState,
)


@pytest.mark.asyncio
async def test_lifecycle_state_starts_in_created_phase() -> None:
    state = AgentExecutionLifecycleState()

    assert state.phase is AgentExecutionLifecyclePhase.CREATED

    snapshot = await state.snapshot()

    assert snapshot.phase is AgentExecutionLifecyclePhase.CREATED
    assert snapshot.sequence == 0
    assert snapshot.transitions == ()


@pytest.mark.asyncio
async def test_lifecycle_state_records_ordered_transitions() -> None:
    state = AgentExecutionLifecycleState()

    first = await state.transition(
        AgentExecutionLifecyclePhase.PREPARING,
        expected_phase=AgentExecutionLifecyclePhase.CREATED,
    )
    second = await state.transition(
        AgentExecutionLifecyclePhase.PRE_EXECUTION,
        expected_phase=AgentExecutionLifecyclePhase.PREPARING,
    )
    third = await state.transition(
        AgentExecutionLifecyclePhase.ORCHESTRATING,
        expected_phase=AgentExecutionLifecyclePhase.PRE_EXECUTION,
    )

    assert first.sequence == 1
    assert first.from_phase is AgentExecutionLifecyclePhase.CREATED
    assert first.to_phase is AgentExecutionLifecyclePhase.PREPARING

    assert second.sequence == 2
    assert third.sequence == 3

    snapshot = await state.snapshot()

    assert snapshot.phase is AgentExecutionLifecyclePhase.ORCHESTRATING
    assert snapshot.sequence == 3
    assert [transition.sequence for transition in snapshot.transitions] == [1, 2, 3]


@pytest.mark.asyncio
async def test_lifecycle_state_rejects_stale_expected_phase() -> None:
    state = AgentExecutionLifecycleState()

    await state.transition(
        AgentExecutionLifecyclePhase.PREPARING,
        expected_phase=AgentExecutionLifecyclePhase.CREATED,
    )

    with pytest.raises(RuntimeError, match="expected phase"):
        await state.transition(
            AgentExecutionLifecyclePhase.ORCHESTRATING,
            expected_phase=AgentExecutionLifecyclePhase.CREATED,
        )

    assert state.phase is AgentExecutionLifecyclePhase.PREPARING


@pytest.mark.asyncio
async def test_lifecycle_state_rejects_duplicate_phase_transition() -> None:
    state = AgentExecutionLifecycleState()

    await state.transition(
        AgentExecutionLifecyclePhase.PREPARING,
        expected_phase=AgentExecutionLifecyclePhase.CREATED,
    )

    with pytest.raises(RuntimeError, match="already"):
        await state.transition(
            AgentExecutionLifecyclePhase.PREPARING,
            expected_phase=AgentExecutionLifecyclePhase.PREPARING,
        )


@pytest.mark.asyncio
async def test_lifecycle_state_transition_is_atomic_under_concurrency() -> None:
    state = AgentExecutionLifecycleState()

    await state.transition(
        AgentExecutionLifecyclePhase.PREPARING,
        expected_phase=AgentExecutionLifecyclePhase.CREATED,
    )

    async def advance() -> bool:
        try:
            await state.transition(
                AgentExecutionLifecyclePhase.PRE_EXECUTION,
                expected_phase=AgentExecutionLifecyclePhase.PREPARING,
            )
        except RuntimeError:
            return False

        return True

    results = await asyncio.gather(advance(), advance())

    assert sorted(results) == [False, True]
    assert state.phase is AgentExecutionLifecyclePhase.PRE_EXECUTION

    snapshot = await state.snapshot()
    assert snapshot.sequence == 2
    assert len(snapshot.transitions) == 2


@pytest.mark.asyncio
async def test_lifecycle_snapshot_is_immutable() -> None:
    state = AgentExecutionLifecycleState()

    await state.transition(
        AgentExecutionLifecyclePhase.PREPARING,
        expected_phase=AgentExecutionLifecyclePhase.CREATED,
    )

    snapshot = await state.snapshot()

    assert isinstance(snapshot.transitions, tuple)

    with pytest.raises(AttributeError):
        snapshot.phase = AgentExecutionLifecyclePhase.FAILED
