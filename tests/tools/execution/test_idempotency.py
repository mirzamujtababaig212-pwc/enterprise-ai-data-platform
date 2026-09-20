from __future__ import annotations

import asyncio

import pytest

from tools.execution.idempotency import (
    InMemoryToolExecutionIdempotencyStore,
    ToolExecutionIdempotencyKey,
    ToolIdempotencyClaimStatus,
)
from tools.models import ToolExecutionResult


def test_idempotency_key_requires_non_empty_fields() -> None:
    with pytest.raises(ValueError):
        ToolExecutionIdempotencyKey(
            run_id="",
            call_id="call-1",
            tool_name="test_tool",
        )

    with pytest.raises(ValueError):
        ToolExecutionIdempotencyKey(
            run_id="run-1",
            call_id="",
            tool_name="test_tool",
        )

    with pytest.raises(ValueError):
        ToolExecutionIdempotencyKey(
            run_id="run-1",
            call_id="call-1",
            tool_name="",
        )


@pytest.mark.asyncio
async def test_first_claim_is_owned_by_caller() -> None:
    store = InMemoryToolExecutionIdempotencyStore()

    key = ToolExecutionIdempotencyKey(
        run_id="run-1",
        call_id="call-1",
        tool_name="test_tool",
    )

    claim = await store.claim(key)

    assert claim.status == ToolIdempotencyClaimStatus.CLAIMED
    assert claim.result is None


@pytest.mark.asyncio
async def test_duplicate_claim_while_execution_is_in_progress_is_rejected() -> None:
    store = InMemoryToolExecutionIdempotencyStore()

    key = ToolExecutionIdempotencyKey(
        run_id="run-1",
        call_id="call-1",
        tool_name="test_tool",
    )

    first = await store.claim(key)
    second = await store.claim(key)

    assert first.status == ToolIdempotencyClaimStatus.CLAIMED
    assert second.status == ToolIdempotencyClaimStatus.IN_PROGRESS
    assert second.result is None


@pytest.mark.asyncio
async def test_completed_result_is_replayed() -> None:
    store = InMemoryToolExecutionIdempotencyStore()

    key = ToolExecutionIdempotencyKey(
        run_id="run-1",
        call_id="call-1",
        tool_name="test_tool",
    )

    result = ToolExecutionResult(
        tool_name="test_tool",
        success=True,
        output={"status": "success"},
    )

    await store.claim(key)
    await store.complete(key, result)

    claim = await store.claim(key)

    assert claim.status == ToolIdempotencyClaimStatus.COMPLETED
    assert claim.result == result


@pytest.mark.asyncio
async def test_failed_execution_can_release_key_for_future_execution() -> None:
    store = InMemoryToolExecutionIdempotencyStore()

    key = ToolExecutionIdempotencyKey(
        run_id="run-1",
        call_id="call-1",
        tool_name="test_tool",
    )

    await store.claim(key)
    await store.release(key)

    claim = await store.claim(key)

    assert claim.status == ToolIdempotencyClaimStatus.CLAIMED


@pytest.mark.asyncio
async def test_different_runs_have_independent_idempotency_keys() -> None:
    store = InMemoryToolExecutionIdempotencyStore()

    key_one = ToolExecutionIdempotencyKey(
        run_id="run-1",
        call_id="call-1",
        tool_name="test_tool",
    )
    key_two = ToolExecutionIdempotencyKey(
        run_id="run-2",
        call_id="call-1",
        tool_name="test_tool",
    )

    first = await store.claim(key_one)
    second = await store.claim(key_two)

    assert first.status == ToolIdempotencyClaimStatus.CLAIMED
    assert second.status == ToolIdempotencyClaimStatus.CLAIMED


@pytest.mark.asyncio
async def test_concurrent_duplicate_claims_have_single_owner() -> None:
    store = InMemoryToolExecutionIdempotencyStore()

    key = ToolExecutionIdempotencyKey(
        run_id="run-1",
        call_id="call-1",
        tool_name="test_tool",
    )

    claims = await asyncio.gather(
        store.claim(key),
        store.claim(key),
        store.claim(key),
    )

    statuses = [claim.status for claim in claims]

    assert statuses.count(ToolIdempotencyClaimStatus.CLAIMED) == 1
    assert statuses.count(ToolIdempotencyClaimStatus.IN_PROGRESS) == 2
