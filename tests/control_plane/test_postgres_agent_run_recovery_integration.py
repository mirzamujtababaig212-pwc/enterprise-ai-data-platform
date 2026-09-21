from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.models import AgentDefinition, AgentRequest, AgentResponse
from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.agents.tool_calls import AgentToolCall
from app.control_plane.persistence.models import (
    AgentRunCheckpointRecord,
    AgentRunRecord,
    ToolExecutionIdempotencyRecord,
)
from app.control_plane.tool_execution.postgres_idempotency import (
    PostgreSQLToolExecutionIdempotencyStore,
)
from sqlalchemy import select, update
from tools.execution.service import ToolExecutionService
from tools.models import ToolDefinition
from tools.registry.in_memory import InMemoryToolRegistry
from ai_platform.agents.observability import AgentExecutionEventType
from ai_platform.agents.llm_messages import user_message
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunStatus,
)
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.agent_run_events.postgres_observer import (
    PostgreSQLAgentRunEventObserver,
)
from app.control_plane.agent_run_events.postgres_repository import (
    PostgreSQLAgentRunEventsRepository,
)
from app.control_plane.agent_runs.recovery_service import (
    AgentRunRecoveryService,
)
from app.control_plane.agent_runs.request_snapshot import (
    AgentRunRequestSnapshot,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = []

    async def resume(
        self,
        agent_name,
        request,
        checkpoint,
        *,
        run_id=None,
        execution_ownership_lost=None,
    ):
        self.calls.append(
            {
                "agent_name": agent_name,
                "request": request,
                "checkpoint": checkpoint,
                "run_id": run_id,
            }
        )

        return AgentResponse(
            agent_name=agent_name,
            output="Recovered from PostgreSQL checkpoint.",
            session_id=request.session_id,
        )


def make_engine():
    host = os.getenv("POSTGRES_TEST_HOST", "localhost")
    port = os.getenv("POSTGRES_TEST_PORT", "5432")
    user = os.getenv("POSTGRES_TEST_USER", "postgres")
    password = os.getenv("POSTGRES_TEST_PASSWORD", "postgres")
    database = os.getenv("POSTGRES_TEST_DB", "vehicle_platform")

    return create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        future=True,
    )


def make_checkpoint(run_id: str) -> AgentExecutionCheckpoint:
    return AgentExecutionCheckpoint(
        schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
        run_id=run_id,
        agent_name="recoverable-agent",
        session_id="session-postgres",
        user_id="user-postgres",
        messages=(user_message("Find vehicle incidents for fleet-42"),),
        tool_round=1,
        position=AgentCheckpointPosition.AFTER_TOOL_EXECUTION,
        metadata={
            "source": "postgres_recovery_integration",
        },
    )


def make_request_snapshot() -> AgentRunRequestSnapshot:
    return AgentRunRequestSnapshot.from_request(
        AgentRequest(
            input="Find vehicle incidents for fleet-42",
            session_id="session-postgres",
            user_id="user-postgres",
            memory_namespace="fleet-memory",
            metadata={
                "request_id": "postgres-recovery-request",
                "source": "integration-test",
            },
        )
    )


class CrashBoundaryTool:
    def __init__(self) -> None:
        self._definition = ToolDefinition(
            name="crash.boundary.tool",
            description="Deterministic side-effect tool for recovery testing.",
        )
        self.execution_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execution_count += 1
        return {
            "status": "side_effect_completed",
            "value": arguments["value"],
            "execution_count": self.execution_count,
        }


class CrashBoundaryLLMGateway:
    def __init__(self) -> None:
        self.requests = []
        self.call_count = 0

    async def route_chat(self, request):
        self.requests.append(request)
        self.call_count += 1

        if self.call_count == 1:
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="crash-boundary-call-001",
                        name="crash.boundary.tool",
                        arguments={"value": "fleet-42"},
                    ),
                ],
            }

        if self.call_count == 2:
            raise RuntimeError("simulated process crash after tool execution")

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "Recovered execution completed successfully.",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
            },
        }


class LeaseLossLLMGateway:
    def __init__(self) -> None:
        self.requests = []
        self.call_count = 0

    async def route_chat(self, request):
        self.requests.append(request)
        self.call_count += 1

        if self.call_count == 1:
            return {
                "provider": "fake",
                "model": request["model"],
                "reply": "",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "tool_calls": [
                    AgentToolCall(
                        call_id="lease-loss-call-001",
                        name="lease.loss.tool",
                        arguments={"value": "fleet-42"},
                    ),
                ],
            }

        return {
            "provider": "fake",
            "model": request["model"],
            "reply": "Recovered without duplicating the external side effect.",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 8,
                "total_tokens": 28,
            },
        }


class LeaseLossTool:
    def __init__(self, ownership_lost) -> None:
        self._definition = ToolDefinition(
            name="lease.loss.tool",
            description="Tool whose external side effect completes before lease loss.",
        )
        self.ownership_lost = ownership_lost
        self.execution_count = 0

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execution_count += 1

        # The external side effect has completed. The durable execution
        # owner is then lost before ToolExecutionService can persist
        # COMPLETED, forcing the outcome into AMBIGUOUS.
        self.ownership_lost.set()

        return {
            "status": "side_effect_completed",
            "value": arguments["value"],
            "execution_count": self.execution_count,
        }


class CrashBoundaryCheckpointHandler:
    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory
        self.suppress_after_tool_checkpoint = True

    async def save(self, checkpoint: AgentExecutionCheckpoint) -> None:
        if (
            self.suppress_after_tool_checkpoint
            and checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
        ):
            return

        session = self._session_factory()
        try:
            PostgreSQLAgentCheckpointsRepository(session).save(checkpoint)
        finally:
            session.close()


def test_postgres_recovery_replays_completed_tool_from_before_checkpoint() -> None:
    engine = make_engine()
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-crash-boundary-recovery-e2e"

    session = session_factory()
    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        checkpoint_repository = PostgreSQLAgentCheckpointsRepository(session)

        request = AgentRequest(
            input="Execute the fleet-42 recovery operation.",
            session_id="session-crash-boundary",
            user_id="user-crash-boundary",
            metadata={
                "request_id": "crash-boundary-request",
                "source": "integration-test",
            },
        )

        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="crash-boundary-agent",
                session_id=request.session_id,
                user_id=request.user_id,
                status=AgentRunStatus.RUNNING,
                started_at=datetime.now(UTC),
                lease_id="initial-lease",
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
                request_snapshot=AgentRunRequestSnapshot.from_request(request),
            )
        )

        tool = CrashBoundaryTool()
        tool_registry = InMemoryToolRegistry()

        import asyncio

        asyncio.run(tool_registry.register(tool))

        idempotency_store = PostgreSQLToolExecutionIdempotencyStore(
            session_factory,
        )
        tool_execution_service = ToolExecutionService(
            tool_registry,
            idempotency_store=idempotency_store,
        )

        llm_gateway = CrashBoundaryLLMGateway()
        registry = InMemoryAgentRegistry()

        definition = AgentDefinition(
            name="crash-boundary-agent",
            description="Crash-boundary recovery integration agent.",
            system_prompt="Execute the requested operation using the available tool.",
            model="fake-model",
            temperature=0.0,
            max_tokens=256,
            tool_names=("crash.boundary.tool",),
        )

        checkpoint_handler = CrashBoundaryCheckpointHandler(session_factory)
        observer = PostgreSQLAgentRunEventObserver(session_factory)

        agent = LLMAgent(
            definition,
            observer=observer,
            checkpoint_handler=checkpoint_handler,
        )

        asyncio.run(registry.register(agent))

        runtime = AgentRuntime(
            registry,
            tool_registry=tool_registry,
            tool_execution_service=tool_execution_service,
            llm_gateway=llm_gateway,
        )

        with pytest.raises(RuntimeError, match="simulated process crash"):
            asyncio.run(
                runtime.run(
                    "crash-boundary-agent",
                    request,
                    run_id=run_id,
                )
            )

        # The tool's side effect completed before the simulated crash.
        assert tool.execution_count == 1
        assert llm_gateway.call_count == 2

        # The AFTER checkpoint was deliberately suppressed, so the durable
        # recovery point is the BEFORE_TOOL_EXECUTION checkpoint.
        checkpoint = checkpoint_repository.get_latest(run_id)
        assert checkpoint is not None
        assert checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
        assert checkpoint.tool_round == 1
        assert checkpoint.execution_budget_state.llm_calls == 1
        assert checkpoint.execution_budget_state.tool_calls == 1
        assert checkpoint.execution_budget_state.tool_rounds == 1

        # Verify the durable idempotency result exists before recovery.
        idempotency_session = session_factory()
        try:
            idempotency_record = idempotency_session.scalar(
                select(ToolExecutionIdempotencyRecord).where(
                    ToolExecutionIdempotencyRecord.run_id == run_id,
                    ToolExecutionIdempotencyRecord.call_id == "crash-boundary-call-001",
                    ToolExecutionIdempotencyRecord.tool_name == "crash.boundary.tool",
                )
            )
            assert idempotency_record is not None
            assert idempotency_record.status == "completed"
            assert idempotency_record.success is True
            assert idempotency_record.output == {
                "status": "side_effect_completed",
                "value": "fleet-42",
                "execution_count": 1,
            }
        finally:
            idempotency_session.close()

        # Simulate the process having disappeared after the tool completed:
        # the run is left RUNNING with an expired lease, while the BEFORE
        # checkpoint and completed idempotency record remain durable.
        expired_at = datetime.now(UTC) - timedelta(minutes=5)
        session.execute(
            update(AgentRunRecord)
            .where(AgentRunRecord.run_id == run_id)
            .values(
                status=AgentRunStatus.RUNNING.value,
                lease_id="expired-after-crash",
                lease_expires_at=expired_at,
            )
        )
        session.commit()

        # Recovery must now allow the AFTER checkpoint to be persisted.
        checkpoint_handler.suppress_after_tool_checkpoint = False

        recovery_service = AgentRunRecoveryService(
            runtime=runtime,
            repository=run_repository,
            checkpoints_repository=checkpoint_repository,
            observer=observer,
            lease_seconds=60,
        )

        results = asyncio.run(
            recovery_service.recover_stale_runs(
                stale_before=datetime.now(UTC),
                limit=10,
            )
        )

        assert len(results) == 1
        assert results[0].run_id == run_id
        assert results[0].response.output == ("Recovered execution completed successfully.")

        # The recovery path reconstructed the original call_id and reached
        # ToolExecutionService, but PostgreSQL idempotency returned the
        # completed result instead of executing the side effect again.
        assert tool.execution_count == 1
        assert llm_gateway.call_count == 3

        session.expire_all()

        restored = run_repository.get(run_id)
        assert restored is not None
        assert restored.status is AgentRunStatus.COMPLETED
        assert restored.output == "Recovered execution completed successfully."
        assert restored.lease_id is None
        assert restored.lease_expires_at is None

        restored_checkpoint = checkpoint_repository.get_latest(run_id)
        assert restored_checkpoint is not None
        assert restored_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION
        assert restored_checkpoint.tool_round == 1
        assert restored_checkpoint.execution_budget_state.llm_calls == 1
        assert restored_checkpoint.execution_budget_state.tool_calls == 1
        assert restored_checkpoint.execution_budget_state.tool_rounds == 1

        # The recovered AFTER checkpoint must contain the replayed tool result.
        assert any(
            message.role.value == "tool" and "crash-boundary-call-001" in message.content
            for message in restored_checkpoint.messages
        )

    finally:
        session.rollback()
        session.execute(
            delete(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunCheckpointRecord).where(
                AgentRunCheckpointRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
        session.close()
        engine.dispose()


def test_postgres_recovery_does_not_repeat_side_effect_after_lease_loss() -> None:
    engine = make_engine()
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "pg-lease-loss-ambiguous-e2e"

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        checkpoint_repository = PostgreSQLAgentCheckpointsRepository(session)

        request = AgentRequest(
            input="Execute the fleet-42 recovery operation.",
            session_id="session-lease-loss",
            user_id="user-lease-loss",
            metadata={
                "request_id": "lease-loss-request",
                "source": "integration-test",
            },
        )

        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="lease-loss-agent",
                session_id=request.session_id,
                user_id=request.user_id,
                status=AgentRunStatus.RUNNING,
                started_at=datetime.now(UTC),
                lease_id="initial-lease",
                lease_expires_at=datetime.now(UTC) + timedelta(minutes=5),
                request_snapshot=AgentRunRequestSnapshot.from_request(request),
            )
        )

        import asyncio

        ownership_lost = asyncio.Event()
        tool = LeaseLossTool(ownership_lost)
        tool_registry = InMemoryToolRegistry()
        asyncio.run(tool_registry.register(tool))

        idempotency_store = PostgreSQLToolExecutionIdempotencyStore(
            session_factory,
        )
        tool_execution_service = ToolExecutionService(
            tool_registry,
            idempotency_store=idempotency_store,
        )

        llm_gateway = LeaseLossLLMGateway()
        registry = InMemoryAgentRegistry()

        definition = AgentDefinition(
            name="lease-loss-agent",
            description="Lease-loss ambiguity recovery integration agent.",
            system_prompt="Execute the requested operation using the available tool.",
            model="fake-model",
            temperature=0.0,
            max_tokens=256,
            tool_names=("lease.loss.tool",),
        )

        checkpoint_handler = CrashBoundaryCheckpointHandler(session_factory)
        observer = PostgreSQLAgentRunEventObserver(session_factory)

        agent = LLMAgent(
            definition,
            observer=observer,
            checkpoint_handler=checkpoint_handler,
        )

        asyncio.run(registry.register(agent))

        runtime = AgentRuntime(
            registry,
            tool_registry=tool_registry,
            tool_execution_service=tool_execution_service,
            llm_gateway=llm_gateway,
        )

        ownership_error = None

        try:
            asyncio.run(
                runtime.run(
                    "lease-loss-agent",
                    request,
                    run_id=run_id,
                    execution_ownership_lost=ownership_lost,
                )
            )
        except Exception as exc:
            ownership_error = exc

        assert ownership_error is not None
        assert ownership_error.__class__.__name__ == ("AgentExecutionOwnershipLostError")

        # The external side effect happened exactly once.
        assert tool.execution_count == 1
        assert llm_gateway.call_count == 1

        # The AFTER_TOOL checkpoint must not have been persisted because
        # ownership was lost after the external side effect.
        checkpoint = checkpoint_repository.get_latest(run_id)
        assert checkpoint is not None
        assert checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION

        # The tool outcome must be durably ambiguous rather than completed
        # or released for another worker to execute.
        idempotency_session = session_factory()
        try:
            idempotency_record = idempotency_session.scalar(
                select(ToolExecutionIdempotencyRecord).where(
                    ToolExecutionIdempotencyRecord.run_id == run_id,
                    ToolExecutionIdempotencyRecord.call_id == "lease-loss-call-001",
                    ToolExecutionIdempotencyRecord.tool_name == "lease.loss.tool",
                )
            )
            assert idempotency_record is not None
            assert idempotency_record.status == "ambiguous"
        finally:
            idempotency_session.close()

        # Simulate worker A disappearing after losing the lease.
        expired_at = datetime.now(UTC) - timedelta(minutes=5)
        session.execute(
            update(AgentRunRecord)
            .where(AgentRunRecord.run_id == run_id)
            .values(
                status=AgentRunStatus.RUNNING.value,
                lease_id="expired-after-lease-loss",
                lease_expires_at=expired_at,
            )
        )
        session.commit()

        checkpoint_handler.suppress_after_tool_checkpoint = False

        recovery_service = AgentRunRecoveryService(
            runtime=runtime,
            repository=run_repository,
            checkpoints_repository=checkpoint_repository,
            observer=observer,
            lease_seconds=60,
        )

        results = asyncio.run(
            recovery_service.recover_stale_runs(
                stale_before=datetime.now(UTC),
                limit=10,
            )
        )

        assert len(results) == 1
        assert results[0].run_id == run_id
        assert results[0].response.output == (
            "Recovered without duplicating the external side effect."
        )

        # Recovery encountered the same logical tool call, but the durable
        # AMBIGUOUS state prevented another external side effect.
        assert tool.execution_count == 1
        assert llm_gateway.call_count == 2

        session.expire_all()

        restored = run_repository.get(run_id)
        assert restored is not None
        assert restored.status is AgentRunStatus.COMPLETED
        assert restored.output == ("Recovered without duplicating the external side effect.")
        assert restored.lease_id is None
        assert restored.lease_expires_at is None

        restored_checkpoint = checkpoint_repository.get_latest(run_id)
        assert restored_checkpoint is not None
        assert restored_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION

        # The ambiguous tool result was carried through recovery; the
        # external tool itself was never executed a second time.
        assert any(
            message.role.value == "tool" and "lease-loss-call-001" in message.content
            for message in restored_checkpoint.messages
        )

    finally:
        session.rollback()
        session.execute(
            delete(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunCheckpointRecord).where(
                AgentRunCheckpointRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
        session.close()
        engine.dispose()


def test_postgres_stale_run_recovers_from_persisted_checkpoint() -> None:
    engine = make_engine()
    session_factory = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )

    run_id = "postgres-stale-recovery-e2e"
    stale_before = datetime.now(UTC)
    expired_at = stale_before - timedelta(minutes=5)

    session = session_factory()

    try:
        run_repository = PostgreSQLAgentRunRepository(session)
        checkpoint_repository = PostgreSQLAgentCheckpointsRepository(session)
        event_repository = PostgreSQLAgentRunEventsRepository(session)

        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="recoverable-agent",
                session_id="session-postgres",
                user_id="user-postgres",
                status=AgentRunStatus.RUNNING,
                started_at=expired_at - timedelta(minutes=1),
                lease_id="expired-lease",
                lease_expires_at=expired_at,
                request_snapshot=make_request_snapshot(),
            )
        )

        checkpoint = make_checkpoint(run_id)
        checkpoint_repository.save(checkpoint)

        runtime = FakeRuntime()

        event_observer = PostgreSQLAgentRunEventObserver(session_factory)

        service = AgentRunRecoveryService(
            runtime=runtime,
            repository=run_repository,
            checkpoints_repository=checkpoint_repository,
            observer=event_observer,
            lease_seconds=60,
        )

        results = __import__("asyncio").run(
            service.recover_stale_runs(
                stale_before=stale_before,
                limit=10,
            )
        )

        assert len(results) == 1
        assert results[0].run_id == run_id
        assert results[0].response.output == ("Recovered from PostgreSQL checkpoint.")

        assert len(runtime.calls) == 1

        call = runtime.calls[0]

        assert call["run_id"] == run_id
        assert call["agent_name"] == "recoverable-agent"
        assert call["checkpoint"] is not None
        assert call["checkpoint"].schema_version == checkpoint.schema_version
        assert call["checkpoint"].run_id == checkpoint.run_id
        assert call["checkpoint"].agent_name == checkpoint.agent_name
        assert call["checkpoint"].session_id == checkpoint.session_id
        assert call["checkpoint"].user_id == checkpoint.user_id
        assert call["checkpoint"].messages == checkpoint.messages
        assert call["checkpoint"].tool_round == checkpoint.tool_round
        assert call["checkpoint"].position == checkpoint.position
        assert call["checkpoint"].metadata == checkpoint.metadata
        assert (
            call["checkpoint"].execution_budget_state.llm_calls
            == checkpoint.execution_budget_state.llm_calls
        )
        assert (
            call["checkpoint"].execution_budget_state.tool_calls
            == checkpoint.execution_budget_state.tool_calls
        )
        assert (
            call["checkpoint"].execution_budget_state.tool_rounds
            == checkpoint.execution_budget_state.tool_rounds
        )

        request = call["request"]

        assert request.input == "Find vehicle incidents for fleet-42"
        assert request.session_id == "session-postgres"
        assert request.user_id == "user-postgres"
        assert request.memory_namespace == "fleet-memory"
        assert request.metadata == {
            "request_id": "postgres-recovery-request",
            "source": "integration-test",
        }

        session.expire_all()

        restored = run_repository.get(run_id)

        assert restored is not None
        assert restored.status is AgentRunStatus.COMPLETED
        assert restored.output == "Recovered from PostgreSQL checkpoint."
        assert restored.lease_id is None
        assert restored.lease_expires_at is None
        assert restored.error_type is None
        assert restored.error_message is None

        restored_checkpoint = checkpoint_repository.get_latest(run_id)

        assert restored_checkpoint is not None
        assert restored_checkpoint.schema_version == checkpoint.schema_version
        assert restored_checkpoint.run_id == checkpoint.run_id
        assert restored_checkpoint.agent_name == checkpoint.agent_name
        assert restored_checkpoint.session_id == checkpoint.session_id
        assert restored_checkpoint.user_id == checkpoint.user_id
        assert restored_checkpoint.messages == checkpoint.messages
        assert restored_checkpoint.tool_round == checkpoint.tool_round
        assert restored_checkpoint.position == checkpoint.position
        assert restored_checkpoint.metadata == checkpoint.metadata
        assert (
            restored_checkpoint.execution_budget_state.llm_calls
            == checkpoint.execution_budget_state.llm_calls
        )
        assert (
            restored_checkpoint.execution_budget_state.tool_calls
            == checkpoint.execution_budget_state.tool_calls
        )
        assert (
            restored_checkpoint.execution_budget_state.tool_rounds
            == checkpoint.execution_budget_state.tool_rounds
        )

        events = event_repository.list(run_id)

        assert [event.event_type for event in events] == [
            AgentExecutionEventType.AGENT_RECOVERY_STARTED,
            AgentExecutionEventType.AGENT_RECOVERY_COMPLETED,
        ]

        assert events[0].agent_name == "recoverable-agent"
        assert events[0].session_id == "session-postgres"
        assert events[0].user_id == "user-postgres"
        assert events[0].metadata == {
            "recovery_type": "stale_run",
        }

        assert events[1].agent_name == "recoverable-agent"
        assert events[1].session_id == "session-postgres"
        assert events[1].user_id == "user-postgres"
        assert events[1].metadata == {
            "recovery_type": "stale_run",
        }

        assert "lease_id" not in events[0].metadata
        assert "lease_id" not in events[1].metadata
    finally:
        session.rollback()
        session.execute(
            delete(AgentRunCheckpointRecord).where(
                AgentRunCheckpointRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
        session.close()
        engine.dispose()
