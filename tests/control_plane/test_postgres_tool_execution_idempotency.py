import asyncio
import os

import pytest
from sqlalchemy import create_engine, delete, inspect, select
from sqlalchemy.orm import sessionmaker

from app.control_plane.persistence.models import (
    Base,
    ToolExecutionIdempotencyRecord,
)
from app.control_plane.tool_execution.postgres_idempotency import (
    PostgreSQLToolExecutionIdempotencyStore,
)
from tools.execution.idempotency import (
    ToolExecutionIdempotencyKey,
    ToolIdempotencyClaimStatus,
)
from tools.models import ToolExecutionFailureCategory, ToolExecutionResult
from tests.tools.execution.test_service import FakeTool
from tools.execution.context import ToolExecutionContext
from tools.execution.service import ToolExecutionService
from tools.registry.in_memory import InMemoryToolRegistry


def make_key(
    *,
    run_id: str = "run-1",
    call_id: str = "call-1",
    tool_name: str = "test_tool",
) -> ToolExecutionIdempotencyKey:
    return ToolExecutionIdempotencyKey(
        run_id=run_id,
        call_id=call_id,
        tool_name=tool_name,
    )


def make_result(
    *,
    tool_name: str = "test_tool",
    output=None,
) -> ToolExecutionResult:
    return ToolExecutionResult(
        tool_name=tool_name,
        success=True,
        output={"value": 42} if output is None else output,
    )


def clear_integration_key(
    session_factory,
    key: ToolExecutionIdempotencyKey,
) -> None:
    session = session_factory()
    try:
        session.execute(
            delete(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == key.run_id,
                ToolExecutionIdempotencyRecord.call_id == key.call_id,
                ToolExecutionIdempotencyRecord.tool_name == key.tool_name,
            )
        )
        session.commit()
    finally:
        session.close()


@pytest.fixture()
def repository():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(engine)

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    try:
        yield engine, session_factory, PostgreSQLToolExecutionIdempotencyStore(session_factory)
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_schema_contains_tool_execution_idempotency_table(
    repository,
) -> None:
    engine, _, _ = repository

    assert inspect(engine).has_table("tool_execution_idempotency")


@pytest.mark.asyncio
async def test_first_claim_is_owned_by_caller(repository) -> None:
    _, _, store = repository

    claim = await store.claim(make_key())

    assert claim.status is ToolIdempotencyClaimStatus.CLAIMED
    assert claim.result is None


@pytest.mark.asyncio
async def test_duplicate_claim_while_in_progress_returns_in_progress(
    repository,
) -> None:
    _, _, store = repository
    key = make_key()

    first = await store.claim(key)
    second = await store.claim(key)

    assert first.status is ToolIdempotencyClaimStatus.CLAIMED
    assert second.status is ToolIdempotencyClaimStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_ambiguous_claim_is_durable(repository) -> None:
    _, session_factory, store = repository
    key = make_key(
        run_id="ambiguous-run",
        call_id="ambiguous-call",
        tool_name="ambiguous_tool",
    )

    claim = await store.claim(key)

    assert claim.status is ToolIdempotencyClaimStatus.CLAIMED
    assert claim.claim_token is not None

    await store.mark_ambiguous(
        key,
        claim_token=claim.claim_token,
    )

    session = session_factory()
    try:
        record = session.scalar(
            select(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == key.run_id,
                ToolExecutionIdempotencyRecord.call_id == key.call_id,
                ToolExecutionIdempotencyRecord.tool_name == key.tool_name,
            )
        )
    finally:
        session.close()

    assert record is not None
    assert record.status == ToolIdempotencyClaimStatus.AMBIGUOUS.value


@pytest.mark.asyncio
async def test_ambiguous_claim_is_not_reclaimable(repository) -> None:
    _, _, store = repository
    key = make_key(
        run_id="ambiguous-run",
        call_id="ambiguous-call",
        tool_name="ambiguous_tool",
    )

    first = await store.claim(key)

    assert first.status is ToolIdempotencyClaimStatus.CLAIMED
    assert first.claim_token is not None

    await store.mark_ambiguous(
        key,
        claim_token=first.claim_token,
    )

    second = await store.claim(key)

    assert second.status is ToolIdempotencyClaimStatus.AMBIGUOUS
    assert second.result is None


@pytest.mark.asyncio
async def test_completed_result_is_replayed(repository) -> None:
    _, session_factory, store = repository
    key = make_key()

    result = make_result()

    claim = await store.claim(key)
    assert claim.claim_token is not None
    await store.complete(
        key,
        result,
        claim_token=claim.claim_token,
    )

    replay = await store.claim(key)

    assert replay.status is ToolIdempotencyClaimStatus.COMPLETED
    assert replay.result == result

    session = session_factory()
    try:
        record = session.scalar(
            select(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == key.run_id,
                ToolExecutionIdempotencyRecord.call_id == key.call_id,
                ToolExecutionIdempotencyRecord.tool_name == key.tool_name,
            )
        )
    finally:
        session.close()

    assert record is not None
    assert record.status == ToolIdempotencyClaimStatus.COMPLETED.value
    assert record.success is True
    assert record.output == {"value": 42}


@pytest.mark.asyncio
async def test_completed_result_replays_across_restarted_tool_service(
    repository,
) -> None:
    _, session_factory, _ = repository

    registry = InMemoryToolRegistry()
    tool = FakeTool()
    await registry.register(tool)

    first_store = PostgreSQLToolExecutionIdempotencyStore(session_factory)
    first_service = ToolExecutionService(
        registry,
        idempotency_store=first_store,
    )

    context = ToolExecutionContext(
        run_id="run-restart-replay",
        call_id="call-restart-replay",
    )

    first = await first_service.execute(
        "test_tool",
        {"value": 42},
        execution_context=context,
    )

    assert first.success is True
    assert tool.execution_count == 1

    # Simulate a process restart: create a new store and service instance.
    second_store = PostgreSQLToolExecutionIdempotencyStore(session_factory)
    second_service = ToolExecutionService(
        registry,
        idempotency_store=second_store,
    )

    replay = await second_service.execute(
        "test_tool",
        {"value": 42},
        execution_context=context,
    )

    assert replay.success is True
    assert replay == first
    assert tool.execution_count == 1


@pytest.mark.asyncio
async def test_release_allows_future_claim(repository) -> None:
    _, _, store = repository
    key = make_key()

    first = await store.claim(key)
    assert first.status is ToolIdempotencyClaimStatus.CLAIMED
    assert first.claim_token is not None

    await store.release(
        key,
        claim_token=first.claim_token,
    )

    second = await store.claim(key)

    assert second.status is ToolIdempotencyClaimStatus.CLAIMED


@pytest.mark.asyncio
async def test_different_keys_are_independent(repository) -> None:
    _, _, store = repository

    first = await store.claim(make_key(call_id="call-1"))
    second = await store.claim(make_key(call_id="call-2"))

    assert first.status is ToolIdempotencyClaimStatus.CLAIMED
    assert second.status is ToolIdempotencyClaimStatus.CLAIMED


@pytest.mark.asyncio
async def test_non_json_output_is_not_persisted(repository) -> None:
    _, _, store = repository
    key = make_key()

    claim = await store.claim(key)
    assert claim.claim_token is not None

    result = make_result(output={"invalid": {1, 2, 3}})

    await store.complete(
        key,
        result,
        claim_token=claim.claim_token,
    )

    replay = await store.claim(key)

    assert replay.status is ToolIdempotencyClaimStatus.CLAIMED
    assert replay.result is None


@pytest.mark.asyncio
async def test_failed_result_releases_claim(repository) -> None:
    _, _, store = repository
    key = make_key()

    claim = await store.claim(key)
    assert claim.claim_token is not None

    result = ToolExecutionResult(
        tool_name=key.tool_name,
        success=False,
        error="execution failed",
        failure_category=ToolExecutionFailureCategory.EXECUTION_ERROR,
    )

    await store.complete(
        key,
        result,
        claim_token=claim.claim_token,
    )

    next_claim = await store.claim(key)

    assert next_claim.status is ToolIdempotencyClaimStatus.CLAIMED


@pytest.fixture()
def postgres_sessions():
    if os.getenv("RUN_POSTGRES_INTEGRATION") != "1":
        pytest.skip("Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test")

    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
    )

    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    try:
        yield engine, session_factory
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_postgres_cross_session_claim_has_single_owner(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    stores = [
        PostgreSQLToolExecutionIdempotencyStore(session_factory),
        PostgreSQLToolExecutionIdempotencyStore(session_factory),
    ]

    key = make_key(
        run_id="postgres-race-run",
        call_id="postgres-race-call",
        tool_name="postgres_race_tool",
    )

    clear_integration_key(session_factory, key)

    first, second = await asyncio.gather(
        stores[0].claim(key),
        stores[1].claim(key),
    )

    statuses = {first.status, second.status}

    assert statuses == {
        ToolIdempotencyClaimStatus.CLAIMED,
        ToolIdempotencyClaimStatus.IN_PROGRESS,
    }

    owner = first if first.status is ToolIdempotencyClaimStatus.CLAIMED else second
    assert owner.claim_token is not None

    owner_store = stores[0] if first.status is ToolIdempotencyClaimStatus.CLAIMED else stores[1]

    await owner_store.release(
        key,
        claim_token=owner.claim_token,
    )


@pytest.mark.asyncio
async def test_postgres_ambiguous_claim_survives_restarted_store(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    key = make_key(
        run_id="postgres-ambiguous-run",
        call_id="postgres-ambiguous-call",
        tool_name="postgres_ambiguous_tool",
    )

    first_store = PostgreSQLToolExecutionIdempotencyStore(session_factory)

    # Ensure the test can be rerun against the same integration database.
    clear_integration_key(session_factory, key)

    first = await first_store.claim(key)

    assert first.status is ToolIdempotencyClaimStatus.CLAIMED
    assert first.claim_token is not None

    await first_store.mark_ambiguous(
        key,
        claim_token=first.claim_token,
    )

    # Simulate a process restart: the second store has no in-memory state.
    second_store = PostgreSQLToolExecutionIdempotencyStore(session_factory)

    second = await second_store.claim(key)

    assert second.status is ToolIdempotencyClaimStatus.AMBIGUOUS
    assert second.result is None


@pytest.mark.asyncio
async def test_postgres_completed_result_replays_across_restarted_store(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    key = make_key(
        run_id="postgres-completed-run",
        call_id="postgres-completed-call",
        tool_name="postgres_completed_tool",
    )

    first_store = PostgreSQLToolExecutionIdempotencyStore(session_factory)

    # Ensure the test can be rerun against the same integration database.
    clear_integration_key(session_factory, key)

    result = make_result(
        tool_name=key.tool_name,
        output={"vehicle_id": "V-100", "status": "processed"},
    )

    first = await first_store.claim(key)

    assert first.status is ToolIdempotencyClaimStatus.CLAIMED
    assert first.claim_token is not None

    await first_store.complete(
        key,
        result,
        claim_token=first.claim_token,
    )

    # Simulate a process restart: the second store has no in-memory state.
    second_store = PostgreSQLToolExecutionIdempotencyStore(session_factory)

    replay = await second_store.claim(key)

    assert replay.status is ToolIdempotencyClaimStatus.COMPLETED
    assert replay.result == result


@pytest.mark.asyncio
async def test_postgres_stale_completion_cannot_complete_released_claim(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    key = make_key(
        run_id="postgres-stale-completion-run",
        call_id="postgres-stale-completion-call",
        tool_name="postgres-stale-completion-tool",
    )

    first_store = PostgreSQLToolExecutionIdempotencyStore(session_factory)
    second_store = PostgreSQLToolExecutionIdempotencyStore(session_factory)

    # Ensure the test can be rerun against the same integration database.
    clear_integration_key(session_factory, key)

    first = await first_store.claim(key)

    assert first.status is ToolIdempotencyClaimStatus.CLAIMED
    assert first.claim_token is not None

    first_token = first.claim_token

    # Worker A loses its claim before completing the external operation.
    await first_store.release(
        key,
        claim_token=first_token,
    )

    # Worker B subsequently acquires the same logical operation.
    second = await second_store.claim(key)

    assert second.status is ToolIdempotencyClaimStatus.CLAIMED
    assert second.claim_token is not None
    assert second.claim_token != first_token

    # Worker A is stale and must not be able to complete Worker B's claim.
    with pytest.raises(RuntimeError, match="claim is no longer owned"):
        await first_store.complete(
            key,
            make_result(
                tool_name=key.tool_name,
                output={"owner": "stale-worker"},
            ),
            claim_token=first_token,
        )

    # Worker B's claim must remain active after the stale completion attempt.
    in_progress = await first_store.claim(key)

    assert in_progress.status is ToolIdempotencyClaimStatus.IN_PROGRESS

    # Worker B can still complete its own claim.
    current_result = make_result(
        tool_name=key.tool_name,
        output={"owner": "current-worker"},
    )

    await second_store.complete(
        key,
        current_result,
        claim_token=second.claim_token,
    )

    # A replay must observe Worker B's durable result.
    replay = await first_store.claim(key)

    assert replay.status is ToolIdempotencyClaimStatus.COMPLETED
    assert replay.result == current_result


@pytest.mark.asyncio
async def test_postgres_mark_run_claims_ambiguous_marks_only_unresolved_claims(
    postgres_sessions,
) -> None:
    _, session_factory = postgres_sessions

    target_run_id = "postgres-reconcile-run"
    other_run_id = "postgres-reconcile-other-run"

    target_key_one = make_key(
        run_id=target_run_id,
        call_id="call-1",
        tool_name="tool-a",
    )
    target_key_two = make_key(
        run_id=target_run_id,
        call_id="call-2",
        tool_name="tool-b",
    )
    other_key = make_key(
        run_id=other_run_id,
        call_id="call-3",
        tool_name="tool-c",
    )

    store = PostgreSQLToolExecutionIdempotencyStore(session_factory)

    for key in (target_key_one, target_key_two, other_key):
        clear_integration_key(session_factory, key)

    first_claim = await store.claim(target_key_one)
    second_claim = await store.claim(target_key_two)
    other_claim = await store.claim(other_key)

    assert first_claim.status is ToolIdempotencyClaimStatus.CLAIMED
    assert second_claim.status is ToolIdempotencyClaimStatus.CLAIMED
    assert other_claim.status is ToolIdempotencyClaimStatus.CLAIMED

    completed_result = make_result(
        tool_name=target_key_one.tool_name,
        output={"status": "already-completed"},
    )

    await store.complete(
        target_key_one,
        completed_result,
        claim_token=first_claim.claim_token,
    )

    marked = await store.mark_run_claims_ambiguous(target_run_id)

    assert marked == 1

    completed_claim = await store.claim(target_key_one)
    unresolved_claim = await store.claim(target_key_two)
    other_run_claim = await store.claim(other_key)

    assert completed_claim.status is ToolIdempotencyClaimStatus.COMPLETED
    assert completed_claim.result == completed_result

    assert unresolved_claim.status is ToolIdempotencyClaimStatus.AMBIGUOUS
    assert other_run_claim.status is ToolIdempotencyClaimStatus.IN_PROGRESS
