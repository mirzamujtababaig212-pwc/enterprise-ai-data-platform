from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai_platform.agents.models import AgentDefinition, AgentRequest
from app.control_plane.agent_delegation.models import AgentDelegationRequest
from app.control_plane.agent_delegation.policy import AgentDelegationPolicy
from app.control_plane.agent_delegation.service import AgentDelegationService
from app.control_plane.agent_run_steps.in_memory import (
    InMemoryAgentRunStepsRepository,
)
from app.control_plane.agent_runs.in_memory import InMemoryAgentRunRepository
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus


@dataclass
class FakeAgent:
    _definition: AgentDefinition

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    async def run(self, context):
        raise AssertionError("child execution must not occur in Phase 2")


@pytest.fixture
def registry():
    from ai_platform.agents.registry.in_memory import InMemoryAgentRegistry

    return InMemoryAgentRegistry()


def make_parent(
    *,
    run_id: str = "root-run",
    root_run_id: str | None = None,
    parent_run_id: str | None = None,
    parent_step_id: str | None = None,
) -> AgentRun:
    return AgentRun(
        run_id=run_id,
        agent_name="parent-agent",
        root_run_id=root_run_id or run_id,
        parent_run_id=parent_run_id,
        parent_step_id=parent_step_id,
        user_id="user-1",
        tenant_id="tenant-1",
        status=AgentRunStatus.RUNNING,
        metadata={},
    )


def make_request(
    *,
    parent_run_id: str = "root-run",
    idempotency_key: str | None = None,
):
    return AgentDelegationRequest(
        parent_run_id=parent_run_id,
        child_agent_name="specialist-agent",
        child_request=AgentRequest(
            input="Analyze the vehicle telemetry.",
            user_id="user-1",
            tenant_id="tenant-1",
            principal="principal-1",
        ),
        idempotency_key=idempotency_key,
    )


@pytest.mark.asyncio
async def test_creates_durable_child_run_and_parent_step(registry):
    await registry.register(
        FakeAgent(
            AgentDefinition(
                name="specialist-agent",
                description="Specialist",
                system_prompt="Analyze the supplied information.",
            )
        )
    )

    runs = InMemoryAgentRunRepository()
    runs.create(make_parent())

    steps = InMemoryAgentRunStepsRepository()

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=lambda: steps,
    )

    result = await service.delegate(make_request())

    assert result.created is True
    assert result.child_run.agent_name == "specialist-agent"
    assert result.child_run.root_run_id == "root-run"
    assert result.child_run.parent_run_id == "root-run"
    assert result.child_run.parent_step_id == result.parent_step_id
    assert result.child_run.causation_id == (f"root-run:{result.parent_step_id}")
    assert result.child_run.status is AgentRunStatus.PENDING
    assert result.child_run.request_snapshot is not None

    step = steps.get("root-run", result.parent_step_id)

    assert step is not None
    assert step.step_type == "delegation"
    assert step.status is not None
    assert step.status.value == "planned"


@pytest.mark.asyncio
async def test_delegation_is_idempotent(registry):
    await registry.register(
        FakeAgent(
            AgentDefinition(
                name="specialist-agent",
                description="Specialist",
                system_prompt="Analyze the supplied information.",
            )
        )
    )

    runs = InMemoryAgentRunRepository()
    runs.create(make_parent())

    steps = InMemoryAgentRunStepsRepository()

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=lambda: steps,
    )

    first = await service.delegate(make_request(idempotency_key="delegation-1"))
    second = await service.delegate(make_request(idempotency_key="delegation-1"))

    assert first.created is True
    assert second.created is False
    assert second.child_run.run_id == first.child_run.run_id
    assert second.parent_step_id == first.parent_step_id


@pytest.mark.asyncio
async def test_rejects_unknown_child_agent(registry):
    runs = InMemoryAgentRunRepository()
    runs.create(make_parent())

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=InMemoryAgentRunStepsRepository,
    )

    with pytest.raises(ValueError, match="does not exist"):
        await service.delegate(make_request())


@pytest.mark.asyncio
async def test_rejects_disabled_child_agent(registry):
    await registry.register(
        FakeAgent(
            AgentDefinition(
                name="specialist-agent",
                description="Specialist",
                system_prompt="Analyze the supplied information.",
                enabled=False,
            )
        )
    )

    runs = InMemoryAgentRunRepository()
    runs.create(make_parent())

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=InMemoryAgentRunStepsRepository,
    )

    with pytest.raises(ValueError, match="disabled"):
        await service.delegate(make_request())


@pytest.mark.asyncio
async def test_enforces_allowed_child_agents(registry):
    await registry.register(
        FakeAgent(
            AgentDefinition(
                name="specialist-agent",
                description="Specialist",
                system_prompt="Analyze the supplied information.",
            )
        )
    )

    runs = InMemoryAgentRunRepository()
    runs.create(make_parent())

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=InMemoryAgentRunStepsRepository,
        policy=AgentDelegationPolicy(allowed_child_agents=frozenset({"other-agent"})),
    )

    with pytest.raises(PermissionError, match="not permitted"):
        await service.delegate(make_request())


@pytest.mark.asyncio
async def test_enforces_delegation_depth_from_durable_parent_chain(registry):
    await registry.register(
        FakeAgent(
            AgentDefinition(
                name="specialist-agent",
                description="Specialist",
                system_prompt="Analyze the supplied information.",
            )
        )
    )

    runs = InMemoryAgentRunRepository()

    root = make_parent(run_id="root-run")
    child = make_parent(
        run_id="child-run",
        root_run_id="root-run",
        parent_run_id="root-run",
        parent_step_id="root-delegation-step",
    )

    runs.create(root)
    runs.create(child)

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=InMemoryAgentRunStepsRepository,
        policy=AgentDelegationPolicy(max_depth=1),
    )

    with pytest.raises(PermissionError, match="exceeds"):
        await service.delegate(make_request(parent_run_id="child-run"))


@pytest.mark.asyncio
async def test_rejects_child_identity_override(registry):
    await registry.register(
        FakeAgent(
            AgentDefinition(
                name="specialist-agent",
                description="Specialist",
                system_prompt="Analyze the supplied information.",
            )
        )
    )

    runs = InMemoryAgentRunRepository()
    runs.create(make_parent())

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=InMemoryAgentRunStepsRepository,
    )

    request = AgentDelegationRequest(
        parent_run_id="root-run",
        child_agent_name="specialist-agent",
        child_request=AgentRequest(
            input="Analyze the vehicle telemetry.",
            user_id="different-user",
            tenant_id="tenant-1",
        ),
    )

    with pytest.raises(
        ValueError,
        match="identity must inherit",
    ):
        await service.delegate(request)


@pytest.mark.asyncio
async def test_inherits_parent_identity(registry):
    await registry.register(
        FakeAgent(
            AgentDefinition(
                name="specialist-agent",
                description="Specialist",
                system_prompt="Analyze the supplied information.",
            )
        )
    )

    parent = AgentRun(
        run_id="root-run",
        agent_name="parent-agent",
        root_run_id="root-run",
        session_id="parent-session",
        user_id="user-1",
        principal="principal-1",
        tenant_id="tenant-1",
        status=AgentRunStatus.RUNNING,
    )

    runs = InMemoryAgentRunRepository()
    runs.create(parent)

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=InMemoryAgentRunStepsRepository,
    )

    request = AgentDelegationRequest(
        parent_run_id="root-run",
        child_agent_name="specialist-agent",
        child_request=AgentRequest(
            input="Analyze the vehicle telemetry.",
        ),
    )

    result = await service.delegate(request)

    assert result.child_run.user_id == "user-1"
    assert result.child_run.principal == "principal-1"
    assert result.child_run.tenant_id == "tenant-1"
    assert result.child_run.session_id == "parent-session"
