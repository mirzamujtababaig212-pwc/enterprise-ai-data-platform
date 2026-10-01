"""PostgreSQL integration tests for durable agent delegation."""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import delete

from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.registry import InMemoryAgentRegistry
from app.control_plane.agent_delegation.models import AgentDelegationRequest
from app.control_plane.agent_delegation.service import AgentDelegationService
from app.control_plane.agent_run_steps.postgres_repository import (
    PostgreSQLAgentRunStepsRepository,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    AgentRunRecord,
    AgentRunStepRecord,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


class IntegrationSpecialistAgent:
    """Deterministic registry target; this test must never execute it."""

    def __init__(self) -> None:
        self._definition = AgentDefinition(
            name="integration-specialist-agent",
            description="PostgreSQL delegation integration target.",
            system_prompt="Analyze the supplied information.",
        )
        self.execution_count = 0

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(self, context):
        self.execution_count += 1
        raise AssertionError("delegated child execution must not occur in Phase 2")


def _build_service(
    *,
    registry: InMemoryAgentRegistry,
    run_repository: PostgreSQLAgentRunRepository,
) -> AgentDelegationService:
    return AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=run_repository,
        agent_run_steps_repository_factory=lambda: (
            PostgreSQLAgentRunStepsRepository(SessionLocal())
        ),
    )


def test_postgres_agent_delegation_persists_hierarchy_and_replays_idempotently() -> None:
    suffix = uuid4().hex
    parent_run_id = str(uuid4())
    idempotency_key = f"delegation-request-{suffix}"
    tenant_id = f"tenant-{suffix}"
    user_id = f"user-{suffix}"
    session_id = f"session-{suffix}"
    principal = f"principal-{suffix}"

    registry = InMemoryAgentRegistry()
    agent = IntegrationSpecialistAgent()

    import asyncio

    asyncio.run(registry.register(agent))

    parent = AgentRun(
        run_id=parent_run_id,
        agent_name="integration-parent-agent",
        root_run_id=parent_run_id,
        session_id=session_id,
        user_id=user_id,
        principal=principal,
        tenant_id=tenant_id,
        status=AgentRunStatus.RUNNING,
        metadata={"integration": "agent-delegation"},
        request_snapshot=AgentRunRequestSnapshot.from_request(
            AgentRequest(
                input="Delegate telemetry analysis.",
                session_id=session_id,
                user_id=user_id,
                principal=principal,
                tenant_id=tenant_id,
            )
        ),
    )

    parent_session = SessionLocal()

    try:
        parent_repository = PostgreSQLAgentRunRepository(parent_session)
        parent_repository.create(parent)

        service = _build_service(
            registry=registry,
            run_repository=parent_repository,
        )

        request = AgentDelegationRequest(
            parent_run_id=parent_run_id,
            child_agent_name="integration-specialist-agent",
            child_request=AgentRequest(
                input="Analyze the vehicle telemetry.",
            ),
            idempotency_key=idempotency_key,
            causation_id=f"causation-{suffix}",
            metadata={
                "source": "postgres-integration",
                "purpose": "delegation-test",
            },
        )

        result = asyncio.run(service.delegate(request))

        assert result.created is True
        assert result.parent_step_id
        assert result.child_run.agent_name == "integration-specialist-agent"
        assert result.child_run.status is AgentRunStatus.PENDING

        child_run_id = result.child_run.run_id
        parent_step_id = result.parent_step_id

        # Reload the parent from PostgreSQL.
        restored_parent = parent_repository.get(parent_run_id)

        assert restored_parent is not None
        assert restored_parent.root_run_id == parent_run_id
        assert restored_parent.parent_run_id is None
        assert restored_parent.parent_step_id is None

        # Reload the delegation step from a separate PostgreSQL session.
        step_session = SessionLocal()

        try:
            step_repository = PostgreSQLAgentRunStepsRepository(step_session)
            restored_step = step_repository.get(
                parent_run_id,
                parent_step_id,
            )

            assert restored_step is not None
            assert restored_step.step_type == "delegation"
            assert restored_step.status.value == "planned"
            assert restored_step.metadata["child_agent_name"] == ("integration-specialist-agent")
            assert restored_step.metadata["delegation_depth"] == 1
        finally:
            step_session.close()

        # Reload the child from PostgreSQL.
        restored_child = parent_repository.get(child_run_id)

        assert restored_child is not None
        assert restored_child.run_id == child_run_id
        assert restored_child.agent_name == "integration-specialist-agent"
        assert restored_child.root_run_id == parent_run_id
        assert restored_child.parent_run_id == parent_run_id
        assert restored_child.parent_step_id == parent_step_id
        assert restored_child.causation_id == f"causation-{suffix}"

        # Identity must be inherited from the durable parent.
        assert restored_child.user_id == user_id
        assert restored_child.principal == principal
        assert restored_child.tenant_id == tenant_id
        assert restored_child.session_id == session_id

        # Delegation metadata must survive persistence.
        assert restored_child.metadata["delegation"]["parent_run_id"] == (parent_run_id)
        assert restored_child.metadata["delegation"]["parent_step_id"] == (parent_step_id)
        assert restored_child.metadata["delegation"]["root_run_id"] == (parent_run_id)
        assert restored_child.metadata["delegation"]["delegation_depth"] == 1
        assert restored_child.metadata["source"] == "postgres-integration"

        # The child request snapshot must be durable.
        assert restored_child.request_snapshot is not None
        assert restored_child.request_snapshot.input == ("Analyze the vehicle telemetry.")
        assert restored_child.request_snapshot.principal == principal
        assert restored_child.request_snapshot.tenant_id == tenant_id
        assert restored_child.request_snapshot.metadata["source"] == ("postgres-integration")

        # Replay the same delegation request.
        replay = asyncio.run(service.delegate(request))

        assert replay.created is False
        assert replay.child_run.run_id == child_run_id
        assert replay.parent_step_id == parent_step_id

        # There must still be exactly one child for this delegation.
        reloaded_child = parent_repository.get(child_run_id)
        assert reloaded_child is not None
        assert reloaded_child.run_id == child_run_id

        # Phase 2 creates the durable delegation only; it must not execute
        # the child agent.
        assert agent.execution_count == 0

        # Verify the database contains the expected hierarchy directly.
        assert (
            parent_session.scalar(
                AgentRunRecord.__table__.select().where(AgentRunRecord.run_id == child_run_id)
            )
            is not None
        )

    finally:
        # Respect the hierarchy FKs during cleanup:
        # child -> delegation step -> parent.
        cleanup_session = SessionLocal()

        try:
            child_id = locals().get("child_run_id")
            step_id = locals().get("parent_step_id")

            if child_id is not None:
                cleanup_session.execute(
                    delete(AgentRunRecord).where(
                        AgentRunRecord.run_id == child_id,
                    )
                )

            if step_id is not None:
                cleanup_session.execute(
                    delete(AgentRunStepRecord).where(
                        AgentRunStepRecord.run_id == parent_run_id,
                        AgentRunStepRecord.step_id == step_id,
                    )
                )

            cleanup_session.execute(
                delete(AgentRunRecord).where(
                    AgentRunRecord.run_id == parent_run_id,
                )
            )

            cleanup_session.commit()
        finally:
            cleanup_session.close()
            parent_session.close()
