import asyncio
import os

import pytest
from sqlalchemy import create_engine, inspect, select
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
async def test_completed_result_is_replayed(repository) -> None:
    _, session_factory, store = repository
    key = make_key()

    result = make_result()

    await store.claim(key)
    await store.complete(key, result)

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
async def test_release_allows_future_claim(repository) -> None:
    _, _, store = repository
    key = make_key()

    first = await store.claim(key)
    assert first.status is ToolIdempotencyClaimStatus.CLAIMED

    await store.release(key)

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

    await store.claim(key)

    result = make_result(output={"invalid": {1, 2, 3}})

    await store.complete(key, result)

    replay = await store.claim(key)

    assert replay.status is ToolIdempotencyClaimStatus.CLAIMED
    assert replay.result is None


@pytest.mark.asyncio
async def test_failed_result_releases_claim(repository) -> None:
    _, _, store = repository
    key = make_key()

    await store.claim(key)

    result = ToolExecutionResult(
        tool_name=key.tool_name,
        success=False,
        error="execution failed",
        failure_category=ToolExecutionFailureCategory.EXECUTION_ERROR,
    )

    await store.complete(key, result)

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

    await stores[0].release(key)

    first, second = await asyncio.gather(
        stores[0].claim(key),
        stores[1].claim(key),
    )

    statuses = {first.status, second.status}

    assert statuses == {
        ToolIdempotencyClaimStatus.CLAIMED,
        ToolIdempotencyClaimStatus.IN_PROGRESS,
    }

    await stores[0].release(key)
    await stores[1].release(key)
