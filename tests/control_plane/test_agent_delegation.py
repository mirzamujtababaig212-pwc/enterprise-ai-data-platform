from __future__ import annotations

from unittest.mock import AsyncMock, Mock
from dataclasses import dataclass

import pytest

from ai_platform.agents.exceptions import AgentExecutionWaitingForApprovalError
from ai_platform.agents.models import AgentDefinition, AgentRequest
from ai_platform.agents.registry import InMemoryAgentRegistry
from app.control_plane.agent_delegation.models import AgentDelegationRequest
from app.control_plane.agent_delegation.policy import AgentDelegationPolicy
from app.control_plane.agent_delegation.service import AgentDelegationService
from app.control_plane.agent_run_steps.in_memory import (
    InMemoryAgentRunStepsRepository,
)
from app.control_plane.agent_run_steps.models import AgentRunStepStatus
from app.control_plane.agent_runs.in_memory import InMemoryAgentRunRepository
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.exceptions import AgentRunAlreadyExecutingError


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


@pytest.mark.asyncio
async def test_execute_delegation_completes_child_and_parent_step(registry):
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

    class FakeAgentRunApplicationService:
        def __init__(self) -> None:
            self.calls = []

        async def execute_existing_run(
            self,
            *,
            run_id,
            agent_name,
            request,
        ):
            self.calls.append(
                {
                    "run_id": run_id,
                    "agent_name": agent_name,
                    "request": request,
                }
            )

            child = runs.get(run_id)
            assert child is not None

            running = child.transition_to(AgentRunStatus.RUNNING)
            completed = running.transition_to(AgentRunStatus.COMPLETED).model_copy(
                update={
                    "output": "Specialist analysis complete.",
                }
            )
            runs.update(completed)

    application_service = FakeAgentRunApplicationService()

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=lambda: steps,
        agent_run_application_service=application_service,
    )

    result = await service.execute_delegation(make_request())

    assert len(application_service.calls) == 1

    call = application_service.calls[0]
    assert call["run_id"] == result.child_run.run_id
    assert call["agent_name"] == "specialist-agent"
    assert call["request"].input == "Analyze the vehicle telemetry."

    child = runs.get(result.child_run.run_id)
    assert child is not None
    assert child.status is AgentRunStatus.COMPLETED
    assert child.output == "Specialist analysis complete."

    parent_step = steps.get("root-run", result.parent_step_id)
    assert parent_step is not None
    assert parent_step.status is AgentRunStepStatus.COMPLETED
    assert parent_step.output == "Specialist analysis complete."


@pytest.mark.asyncio
async def test_execute_delegation_preserves_running_child_and_parent_step():
    """A concurrently executing child must remain in-flight, not fail delegation."""
    agent_registry = InMemoryAgentRegistry()

    await agent_registry.register(
        FakeAgent(
            AgentDefinition(
                name="child-agent",
                description="Child agent",
                system_prompt="You are a child agent.",
                model="test-model",
                enabled=True,
            )
        )
    )

    repository = InMemoryAgentRunRepository()
    steps_repository = InMemoryAgentRunStepsRepository()

    parent_run = AgentRun(
        run_id="parent-running-child",
        agent_name="parent-agent",
        status=AgentRunStatus.RUNNING,
        user_id="user-1",
        principal="principal-1",
        tenant_id="tenant-1",
        session_id="session-1",
    )
    repository.create(parent_run)

    service = AgentDelegationService(
        agent_registry=agent_registry,
        agent_run_repository=repository,
        agent_run_steps_repository_factory=lambda: steps_repository,
        agent_run_application_service=object(),
    )

    result = await service.delegate(
        AgentDelegationRequest(
            parent_run_id=parent_run.run_id,
            child_agent_name="child-agent",
            child_request=AgentRequest(
                input="child work",
                session_id="session-1",
                user_id="user-1",
                principal="principal-1",
                tenant_id="tenant-1",
            ),
            idempotency_key="running-child",
        )
    )

    child = repository.get(result.child_run.run_id)
    assert child is not None

    child = child.model_copy(
        update={
            "status": AgentRunStatus.RUNNING,
        }
    )
    repository.update(child)

    execution_service = AgentDelegationService(
        agent_registry=agent_registry,
        agent_run_repository=repository,
        agent_run_steps_repository_factory=lambda: steps_repository,
        agent_run_application_service=object(),
    )

    execution_result = await execution_service.execute_delegation(
        AgentDelegationRequest(
            parent_run_id=parent_run.run_id,
            child_agent_name="child-agent",
            child_request=AgentRequest(
                input="child work",
                session_id="session-1",
                user_id="user-1",
                principal="principal-1",
                tenant_id="tenant-1",
            ),
            idempotency_key="running-child",
        )
    )

    assert execution_result.child_run.status is AgentRunStatus.RUNNING

    parent_step = steps_repository.get(
        parent_run.run_id,
        execution_result.parent_step_id,
    )
    assert parent_step is not None
    assert parent_step.status is AgentRunStepStatus.RUNNING


@pytest.mark.asyncio
async def test_execute_delegation_handles_agent_run_already_executing_error():
    """A child claimed concurrently must remain RUNNING and keep its parent step RUNNING."""
    agent_registry = InMemoryAgentRegistry()

    await agent_registry.register(
        FakeAgent(
            AgentDefinition(
                name="child-agent",
                description="Child agent",
                system_prompt="You are a child agent.",
                model="test-model",
                enabled=True,
            )
        )
    )

    repository = InMemoryAgentRunRepository()
    steps_repository = InMemoryAgentRunStepsRepository()

    parent_run = AgentRun(
        run_id="parent-race-child",
        agent_name="parent-agent",
        status=AgentRunStatus.RUNNING,
        user_id="user-1",
        principal="principal-1",
        tenant_id="tenant-1",
        session_id="session-1",
    )
    repository.create(parent_run)

    child_run_id = None

    async def execute_existing_run_with_race(
        *,
        run_id,
        agent_name,
        request,
    ):
        nonlocal child_run_id

        child_run_id = run_id

        child = repository.get(run_id)
        assert child is not None
        assert child.status is AgentRunStatus.PENDING

        # Simulate another worker winning the atomic PENDING -> RUNNING claim
        # immediately before this worker's execute_existing_run() proceeds.
        running_child = child.model_copy(
            update={
                "status": AgentRunStatus.RUNNING,
            }
        )
        repository.update(running_child)

        raise AgentRunAlreadyExecutingError(f"Agent run '{run_id}' is already executing.")

    mock_app_service = Mock()
    mock_app_service.execute_existing_run = AsyncMock(side_effect=execute_existing_run_with_race)

    service = AgentDelegationService(
        agent_registry=agent_registry,
        agent_run_repository=repository,
        agent_run_steps_repository_factory=lambda: steps_repository,
        agent_run_application_service=mock_app_service,
    )

    result = await service.execute_delegation(
        AgentDelegationRequest(
            parent_run_id=parent_run.run_id,
            child_agent_name="child-agent",
            child_request=AgentRequest(
                input="child work",
                session_id="session-1",
                user_id="user-1",
                principal="principal-1",
                tenant_id="tenant-1",
            ),
            idempotency_key="race-child",
        )
    )

    assert child_run_id == result.child_run.run_id
    assert mock_app_service.execute_existing_run.await_count == 1

    child = repository.get(result.child_run.run_id)
    assert child is not None
    assert child.status is AgentRunStatus.RUNNING

    assert result.child_run.status is AgentRunStatus.RUNNING

    parent_step = steps_repository.get(
        parent_run.run_id,
        result.parent_step_id,
    )
    assert parent_step is not None
    assert parent_step.status is AgentRunStepStatus.RUNNING


@pytest.mark.asyncio
async def test_execute_delegation_preserves_running_parent_step_when_child_waits_for_approval(
    registry,
):
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

    class FakeAgentRunApplicationService:
        async def execute_existing_run(
            self,
            *,
            run_id,
            agent_name,
            request,
        ):
            child = runs.get(run_id)
            assert child is not None

            running = child.transition_to(AgentRunStatus.RUNNING)
            waiting = running.transition_to(AgentRunStatus.WAITING_FOR_APPROVAL)
            runs.update(waiting)

            raise AgentExecutionWaitingForApprovalError(
                "Agent execution is waiting for human approval."
            )

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=lambda: steps,
        agent_run_application_service=FakeAgentRunApplicationService(),
    )

    result = await service.execute_delegation(make_request())

    child = runs.get(result.child_run.run_id)
    assert child is not None
    assert child.status is AgentRunStatus.WAITING_FOR_APPROVAL

    parent_step = steps.get("root-run", result.parent_step_id)
    assert parent_step is not None
    assert parent_step.status is AgentRunStepStatus.RUNNING
    assert parent_step.error is None
    assert parent_step.completed_at is None


@pytest.mark.asyncio
async def test_execute_delegation_fails_parent_step_when_child_fails(registry):
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

    class FakeAgentRunApplicationService:
        def __init__(self) -> None:
            self.child_run_id = None

        async def execute_existing_run(
            self,
            *,
            run_id,
            agent_name,
            request,
        ):
            self.child_run_id = run_id
            child = runs.get(run_id)
            assert child is not None

            running = child.transition_to(AgentRunStatus.RUNNING)
            failed = running.transition_to(AgentRunStatus.FAILED).model_copy(
                update={
                    "error_type": "RuntimeError",
                    "error_message": "Child process crashed",
                }
            )
            runs.update(failed)

            raise RuntimeError("Child process crashed")

    application_service = FakeAgentRunApplicationService()

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=lambda: steps,
        agent_run_application_service=application_service,
    )

    with pytest.raises(RuntimeError, match="Child process crashed"):
        await service.execute_delegation(make_request())

    assert application_service.child_run_id is not None

    delegated_child = runs.get(application_service.child_run_id)
    assert delegated_child is not None

    assert delegated_child.status is AgentRunStatus.FAILED
    assert delegated_child.error_type == "RuntimeError"
    assert delegated_child.error_message == "Child process crashed"

    delegation_steps = [step for step in steps.list("root-run") if step.step_type == "delegation"]
    assert len(delegation_steps) == 1

    parent_step = delegation_steps[0]
    assert parent_step.status is AgentRunStepStatus.FAILED
    assert parent_step.error == "Child process crashed"
    assert parent_step.failure_category == "RuntimeError"


@pytest.mark.asyncio
async def test_execute_delegation_preserves_child_lineage(registry):
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

    class FakeAgentRunApplicationService:
        async def execute_existing_run(
            self,
            *,
            run_id,
            agent_name,
            request,
        ):
            child = runs.get(run_id)
            assert child is not None

            running = child.transition_to(AgentRunStatus.RUNNING)
            completed = running.transition_to(AgentRunStatus.COMPLETED).model_copy(
                update={
                    "output": "Lineage preserved.",
                }
            )
            runs.update(completed)

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=lambda: steps,
        agent_run_application_service=FakeAgentRunApplicationService(),
    )

    result = await service.execute_delegation(make_request(idempotency_key="lineage-test"))

    child = runs.get(result.child_run.run_id)
    assert child is not None

    assert child.root_run_id == "root-run"
    assert child.parent_run_id == "root-run"
    assert child.parent_step_id == result.parent_step_id
    assert child.causation_id == (f"root-run:{result.parent_step_id}")
    assert child.status is AgentRunStatus.COMPLETED

    parent_step = steps.get("root-run", result.parent_step_id)
    assert parent_step is not None
    assert parent_step.status is AgentRunStepStatus.COMPLETED


@pytest.mark.asyncio
async def test_execute_delegation_reconciles_already_completed_child(registry):
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

    class FakeAgentRunApplicationService:
        async def execute_existing_run(self, **kwargs):
            raise AssertionError("terminal child should not be executed again")

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=lambda: steps,
        agent_run_application_service=FakeAgentRunApplicationService(),
    )

    result = await service.delegate(make_request(idempotency_key="completed-child"))

    child = runs.get(result.child_run.run_id)
    assert child is not None

    running = child.transition_to(AgentRunStatus.RUNNING)
    completed = running.transition_to(AgentRunStatus.COMPLETED).model_copy(
        update={"output": "Already completed."}
    )
    runs.update(completed)

    execution_result = await service.execute_delegation(
        make_request(idempotency_key="completed-child")
    )

    assert execution_result.child_run.status is AgentRunStatus.COMPLETED

    parent_step = steps.get("root-run", result.parent_step_id)
    assert parent_step is not None
    assert parent_step.status is AgentRunStepStatus.COMPLETED
    assert parent_step.output == "Already completed."


@pytest.mark.asyncio
async def test_execute_delegation_preserves_existing_waiting_child(registry):
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

    class FakeAgentRunApplicationService:
        async def execute_existing_run(self, **kwargs):
            raise AssertionError("waiting child should not be executed again")

    service = AgentDelegationService(
        agent_registry=registry,
        agent_run_repository=runs,
        agent_run_steps_repository_factory=lambda: steps,
        agent_run_application_service=FakeAgentRunApplicationService(),
    )

    result = await service.delegate(make_request(idempotency_key="waiting-child"))

    child = runs.get(result.child_run.run_id)
    assert child is not None

    running = child.transition_to(AgentRunStatus.RUNNING)
    waiting = running.transition_to(AgentRunStatus.WAITING_FOR_APPROVAL)
    runs.update(waiting)

    execution_result = await service.execute_delegation(
        make_request(idempotency_key="waiting-child")
    )

    assert execution_result.child_run.status is AgentRunStatus.WAITING_FOR_APPROVAL

    parent_step = steps.get("root-run", result.parent_step_id)
    assert parent_step is not None
    assert parent_step.status is AgentRunStepStatus.RUNNING
    assert parent_step.completed_at is None
