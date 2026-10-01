from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ai_platform.agents.budget import ExecutionBudget
from ai_platform.agents.exceptions import (
    AgentExecutionOwnershipLostError,
    AgentExecutionWaitingForApprovalError,
)
from ai_platform.agents.models import AgentDefinition, AgentRequest, AgentResponse
from ai_platform.agents.policy import (
    ModelGovernanceDecision,
    TenantPolicy,
    TenantPolicyEngine,
    PolicyViolationError,
)
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)

from app.control_plane.agent_run_events.repository import (
    AgentRunEventsRepository,
)
from app.control_plane.agent_runs.admission import AgentRunAdmissionResult
from app.control_plane.agent_runs.exceptions import (
    AgentRunAccessDeniedError,
    AgentRunAdmissionRejectedError,
    AgentRunIdempotencyConflictError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import (
    AgentRun,
    AgentRunExecutionResult,
    AgentRunStatus,
)
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.agent_runs.application_service import (
    AgentRunApplicationService,
)
from app.control_plane.agent_runs.lease import heartbeat_loop
from app.control_plane.agent_runs.exceptions import AgentRunAlreadyExecutingError
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_steps.repository import (
    AgentRunStepsRepository,
)


def _response(
    *,
    agent_name: str = "enterprise-analyst",
    output="completed",
    session_id: str | None = "session-1",
) -> AgentResponse:
    return AgentResponse(
        agent_name=agent_name,
        output=output,
        session_id=session_id,
        metadata={"source": "test"},
    )


def _repository() -> Mock:
    repository = Mock(spec=AgentRunRepository)
    repository.created_run = None
    repository.running_run = None
    repository.completed_run = None
    repository.failed_run = None
    repository.cancelled_run = None
    repository.waiting_for_approval_run = None

    def _get(run_id: str) -> AgentRun | None:
        for candidate in (
            repository.completed_run,
            repository.failed_run,
            repository.cancelled_run,
            repository.waiting_for_approval_run,
            repository.running_run,
            repository.created_run,
        ):
            if candidate is not None and candidate.run_id == run_id:
                return candidate

        configured = repository.get.return_value
        if isinstance(configured, AgentRun) and configured.run_id == run_id:
            return configured

        return None

    def create_run(run: AgentRun) -> AgentRun:
        repository.created_run = run
        return run

    def claim_pending_run(
        run_id: str,
        *,
        started_at,
        lease_id: str,
        lease_expires_at,
    ) -> AgentRun | None:
        base_run = _get(run_id)

        if base_run is None:
            return None

        if base_run.status is not AgentRunStatus.PENDING:
            return None

        running_run = base_run.model_copy(
            update={
                "status": AgentRunStatus.RUNNING,
                "started_at": started_at,
                "lease_id": lease_id,
                "lease_expires_at": lease_expires_at,
            }
        )
        repository.running_run = running_run
        return running_run

    def complete_if_owner(
        run_id: str,
        *,
        lease_id: str,
        completed_at,
        output,
    ) -> AgentRun:
        running = _get(run_id)
        assert running is not None
        assert running.status is AgentRunStatus.RUNNING

        completed = running.transition_to(
            AgentRunStatus.COMPLETED,
        ).model_copy(
            update={
                "completed_at": completed_at,
                "output": output,
                "lease_id": None,
                "lease_expires_at": None,
            }
        )
        repository.completed_run = completed
        return completed

    def fail_if_owner(
        run_id: str,
        *,
        lease_id: str,
        completed_at,
        error_type: str,
        error_message: str,
        error_details: dict | None = None,
    ) -> AgentRun:
        running = _get(run_id)
        assert running is not None
        assert running.status is AgentRunStatus.RUNNING

        failed = running.transition_to(
            AgentRunStatus.FAILED,
        ).model_copy(
            update={
                "completed_at": completed_at,
                "error_type": error_type,
                "error_message": error_message,
                "error_details": error_details,
                "lease_id": None,
                "lease_expires_at": None,
            }
        )
        repository.failed_run = failed
        return failed

    def cancel_if_owner(
        run_id: str,
        *,
        lease_id: str,
        completed_at,
    ) -> AgentRun:
        running = _get(run_id)
        assert running is not None
        assert running.status is AgentRunStatus.RUNNING

        cancelled = running.transition_to(
            AgentRunStatus.CANCELLED,
        ).model_copy(
            update={
                "completed_at": completed_at,
                "lease_id": None,
                "lease_expires_at": None,
            }
        )
        repository.cancelled_run = cancelled
        return cancelled

    def transition_to_waiting_for_approval_if_owner(
        run_id: str,
        *,
        lease_id: str,
        updated_at,
    ) -> AgentRun:
        running = _get(run_id)
        assert running is not None
        assert running.status is AgentRunStatus.RUNNING

        waiting = running.transition_to(
            AgentRunStatus.WAITING_FOR_APPROVAL,
        ).model_copy(
            update={
                "updated_at": updated_at,
                "lease_id": None,
                "lease_expires_at": None,
            }
        )
        repository.waiting_for_approval_run = waiting
        return waiting

    repository.get.side_effect = _get
    repository.create.side_effect = create_run
    repository.claim_pending_run.side_effect = claim_pending_run
    repository.complete_if_owner.side_effect = complete_if_owner
    repository.fail_if_owner.side_effect = fail_if_owner
    repository.cancel_if_owner.side_effect = cancel_if_owner
    repository.transition_to_waiting_for_approval_if_owner.side_effect = (
        transition_to_waiting_for_approval_if_owner
    )

    return repository


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[AgentExecutionEvent] = []

    async def record(self, event: AgentExecutionEvent) -> None:
        self.events.append(event)


class FailingObserver:
    def __init__(self) -> None:
        self.calls = 0

    async def record(self, event: AgentExecutionEvent) -> None:
        self.calls += 1
        raise RuntimeError("event persistence unavailable")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_max_tokens", "tenant_max_tokens", "expected_max_tokens"),
    [
        (100_000, 20_000, 20_000),
        (10_000, 20_000, 10_000),
        (None, 20_000, 20_000),
    ],
)
async def test_execute_resolves_effective_token_budget_from_tenant_policy(
    requested_max_tokens: int | None,
    tenant_max_tokens: int,
    expected_max_tokens: int,
) -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())
    runtime.get_agent_definition = AsyncMock(
        return_value=AgentDefinition(
            name="enterprise-analyst",
            description="Test enterprise analyst agent.",
            system_prompt="You are a test enterprise analyst.",
            model="gpt-5",
        )
    )

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            max_tokens_per_run=tenant_max_tokens,
        )
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-budget-1",
        user_id="user-budget-1",
        principal="principal-budget-1",
        tenant_id="tenant-acme",
        execution_budget=ExecutionBudget(
            max_tokens_per_run=requested_max_tokens,
        ),
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=request,
    )

    pending = repository.create.call_args.args[0]
    running = repository.running_run

    assert running is not None
    assert pending.request_snapshot.execution_budget["max_tokens_per_run"] == expected_max_tokens
    assert running.request_snapshot.execution_budget["max_tokens_per_run"] == expected_max_tokens

    effective_request = runtime.run.await_args.args[1]

    assert effective_request.execution_budget is not None
    assert effective_request.execution_budget.max_tokens_per_run == expected_max_tokens


@pytest.mark.asyncio
async def test_execute_does_not_record_budget_governance_without_tenant_policy() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        observer=observer,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-budget-no-policy-1",
        user_id="user-budget-no-policy-1",
        principal="principal-budget-no-policy-1",
        execution_budget=ExecutionBudget(
            max_tokens_per_run=10_000,
        ),
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=request,
    )

    budget_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
        and event.metadata.get("governance_domain") == "budget"
    ]

    assert budget_events == []


@pytest.mark.asyncio
async def test_execute_records_effective_budget_governance_decision() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())
    runtime.get_agent_definition = AsyncMock(
        return_value=AgentDefinition(
            name="enterprise-analyst",
            description="Test enterprise analyst agent.",
            system_prompt="You are a test enterprise analyst.",
            model="gpt-5",
        )
    )

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            max_tokens_per_run=20_000,
            policy_id="enterprise-budget-policy",
            policy_version="v3",
        )
    )

    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
        observer=observer,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-budget-governance-1",
        user_id="user-budget-governance-1",
        principal="principal-budget-governance-1",
        tenant_id="tenant-acme",
        execution_budget=ExecutionBudget(
            max_tokens_per_run=100_000,
        ),
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=request,
    )

    pending = repository.create.call_args.args[0]
    assert pending.request_snapshot.execution_budget["max_tokens_per_run"] == 20_000

    budget_events = [
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
        and event.metadata.get("governance_domain") == "budget"
    ]

    assert len(budget_events) == 1

    budget_event = budget_events[0]
    assert budget_event.run_id == pending.run_id
    assert budget_event.metadata == {
        "governance_domain": "budget",
        "decision": "allow",
        "tenant_id": "tenant-acme",
        "policy_id": "enterprise-budget-policy",
        "policy_version": "v3",
        "details": {
            "max_tokens_per_run": 20_000,
        },
    }

    assert "Explain the platform" not in repr(budget_event.metadata)


@pytest.mark.asyncio
async def test_execute_resolves_model_governance_from_tenant_policy() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())
    runtime.get_agent_definition = AsyncMock(
        return_value=AgentDefinition(
            name="enterprise-analyst",
            description="Test enterprise analyst agent.",
            system_prompt="You are a test enterprise analyst.",
            model="gpt-5",
        )
    )

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_models=frozenset({"gpt-5"}),
            policy_id="enterprise-model-policy",
            policy_version="v7",
        )
    )

    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
        observer=observer,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-model-1",
        user_id="user-model-1",
        principal="principal-model-1",
        tenant_id="tenant-acme",
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=request,
    )

    effective_request = runtime.run.await_args.args[1]

    assert isinstance(
        effective_request.model_governance,
        ModelGovernanceDecision,
    )
    assert effective_request.model_governance == ModelGovernanceDecision(
        effective_model="gpt-5",
        policy_id="enterprise-model-policy",
        policy_version="v7",
    )

    pending = repository.create.call_args.args[0]
    running = repository.running_run

    assert running is not None
    assert pending.request_snapshot.model_governance == {
        "effective_model": "gpt-5",
        "effective_provider": None,
        "policy_id": "enterprise-model-policy",
        "policy_version": "v7",
    }
    assert running.request_snapshot.model_governance == (pending.request_snapshot.model_governance)

    assert len(observer.events) == 3

    model_event = observer.events[0]
    assert model_event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert model_event.run_id == pending.run_id
    assert model_event.agent_name == "enterprise-analyst"
    assert model_event.metadata == {
        "governance_domain": "model",
        "decision": "allow",
        "tenant_id": "tenant-acme",
        "policy_id": "enterprise-model-policy",
        "policy_version": "v7",
        "details": {
            "effective_model": "gpt-5",
            "effective_provider": None,
        },
    }

    budget_event = observer.events[1]
    assert budget_event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert budget_event.run_id == pending.run_id
    assert budget_event.metadata == {
        "governance_domain": "budget",
        "decision": "allow",
        "tenant_id": "tenant-acme",
        "policy_id": "enterprise-model-policy",
        "policy_version": "v7",
        "details": {
            "max_tokens_per_run": None,
        },
    }

    admission_event = observer.events[2]
    assert admission_event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert admission_event.metadata == {
        "governance_domain": "admission",
        "decision": "allow",
        "tenant_id": "tenant-acme",
    }

    serialized_event = repr(model_event.metadata)
    assert "Explain the platform" not in serialized_event


@pytest.mark.asyncio
async def test_execute_uses_pinned_model_governance_without_reauthorization() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_models=frozenset({"claude-sonnet-4"}),
            policy_id="current-model-policy",
            policy_version="v8",
        )
    )

    pinned_decision = ModelGovernanceDecision(
        effective_model="gpt-5",
        effective_provider="openai",
        policy_id="enterprise-model-policy",
        policy_version="v7",
    )

    observer = RecordingObserver()

    authorize_model = Mock(wraps=tenant_policy_engine.authorize_model)
    tenant_policy_engine.authorize_model = authorize_model

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
        observer=observer,
    )

    request = AgentRequest(
        input="Resume the platform analysis",
        session_id="session-model-recovery-1",
        user_id="user-model-recovery-1",
        principal="principal-model-recovery-1",
        tenant_id="tenant-acme",
        model_governance=pinned_decision,
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=request,
    )

    runtime.get_agent_definition.assert_not_called()

    effective_request = runtime.run.await_args.args[1]

    tenant_policy_engine.authorize_model.assert_not_called()
    assert effective_request.model_governance is pinned_decision

    pending = repository.create.call_args.args[0]
    assert pending.request_snapshot.model_governance == {
        "effective_model": "gpt-5",
        "effective_provider": "openai",
        "policy_id": "enterprise-model-policy",
        "policy_version": "v7",
    }

    model_event = observer.events[0]
    assert model_event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert model_event.run_id == pending.run_id
    assert model_event.metadata == {
        "governance_domain": "model",
        "decision": "allow",
        "tenant_id": "tenant-acme",
        "policy_id": "enterprise-model-policy",
        "policy_version": "v7",
        "details": {
            "effective_model": "gpt-5",
            "effective_provider": "openai",
        },
    }


@pytest.mark.asyncio
async def test_execute_allows_memory_namespace_authorized_by_tenant_policy() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())
    runtime.get_agent_definition = AsyncMock(
        return_value=AgentDefinition(
            name="enterprise-analyst",
            description="Test enterprise analyst agent.",
            system_prompt="You are a test enterprise analyst.",
            model="gpt-5",
        )
    )

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_memory_namespaces=frozenset({"fleet-42"}),
        )
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-memory-allowed-1",
        user_id="user-memory-allowed-1",
        principal="principal-memory-allowed-1",
        tenant_id="tenant-acme",
        memory_namespace="fleet-42",
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=request,
    )

    assert result.response.output == "completed"
    repository.create.assert_called_once()
    runtime.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_rejects_memory_namespace_not_allowed_by_tenant_policy() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())
    runtime.get_agent_definition = AsyncMock(
        return_value=AgentDefinition(
            name="enterprise-analyst",
            description="Test enterprise analyst agent.",
            system_prompt="You are a test enterprise analyst.",
            model="gpt-5",
        )
    )

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_memory_namespaces=frozenset({"fleet-42"}),
        )
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-memory-denied-1",
        user_id="user-memory-denied-1",
        principal="principal-memory-denied-1",
        tenant_id="tenant-acme",
        memory_namespace="project-beta",
    )

    with pytest.raises(
        PolicyViolationError,
        match="Memory namespace.*not allowed",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=request,
        )

    repository.create.assert_not_called()
    repository.update.assert_not_called()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_authorizes_memory_write_for_memory_writing_agent() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())
    runtime.get_agent_definition = AsyncMock(
        return_value=AgentDefinition(
            name="memory-writer",
            description="Test memory-writing agent.",
            system_prompt="You are a test memory-writing agent.",
            model="gpt-5",
            memory_write_enabled=True,
        )
    )

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_memory_namespaces=frozenset({"fleet-42"}),
        )
    )

    authorize_memory_namespace = Mock(wraps=tenant_policy_engine.authorize_memory_namespace)
    tenant_policy_engine.authorize_memory_namespace = authorize_memory_namespace

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
    )

    request = AgentRequest(
        input="Record the deployment result",
        session_id="session-memory-write-1",
        user_id="user-memory-write-1",
        principal="principal-memory-write-1",
        tenant_id="tenant-acme",
        memory_namespace="fleet-42",
    )

    await service.execute(
        agent_name="memory-writer",
        request=request,
    )

    assert [call.kwargs for call in authorize_memory_namespace.call_args_list] == [
        {
            "tenant_id": "tenant-acme",
            "memory_namespace": "fleet-42",
            "operation": "read",
        },
        {
            "tenant_id": "tenant-acme",
            "memory_namespace": "fleet-42",
            "operation": "write",
        },
    ]
    repository.create.assert_called_once()
    runtime.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_rejects_model_not_allowed_by_tenant_policy() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())
    runtime.get_agent_definition = AsyncMock(
        return_value=AgentDefinition(
            name="enterprise-analyst",
            description="Test enterprise analyst agent.",
            system_prompt="You are a test enterprise analyst.",
            model="gpt-5",
        )
    )

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_models=frozenset({"claude-sonnet-4"}),
            policy_id="enterprise-model-policy",
            policy_version="v7",
        )
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-model-denied-1",
        user_id="user-model-denied-1",
        principal="principal-model-denied-1",
        tenant_id="tenant-acme",
    )

    with pytest.raises(
        PolicyViolationError,
        match="Model 'gpt-5' is not allowed for tenant 'tenant-acme'",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=request,
        )

    repository.create.assert_not_called()
    repository.update.assert_not_called()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_rejects_missing_provider_when_tenant_has_provider_allow_list() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())
    runtime.get_agent_definition = AsyncMock(
        return_value=AgentDefinition(
            name="enterprise-analyst",
            description="Test enterprise analyst agent.",
            system_prompt="You are a test enterprise analyst.",
            model="gpt-5",
        )
    )

    tenant_policy_engine = TenantPolicyEngine()
    tenant_policy_engine.register_policy(
        TenantPolicy(
            tenant_id="tenant-acme",
            allowed_models=frozenset({"gpt-5"}),
            allowed_providers=frozenset({"openai"}),
            policy_id="enterprise-model-policy",
            policy_version="v7",
        )
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-provider-denied-1",
        user_id="user-provider-denied-1",
        principal="principal-provider-denied-1",
        tenant_id="tenant-acme",
    )

    with pytest.raises(
        PolicyViolationError,
        match="No provider was specified for model 'gpt-5'",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=request,
        )

    repository.create.assert_not_called()
    repository.update.assert_not_called()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_rejects_unregistered_tenant_before_run_creation() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    tenant_policy_engine = TenantPolicyEngine()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        tenant_policy_engine=tenant_policy_engine,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-unregistered",
        user_id="user-unregistered",
        principal="principal-unregistered",
        tenant_id="tenant-unregistered",
        execution_budget=ExecutionBudget(
            max_tokens_per_run=10_000,
        ),
    )

    with pytest.raises(
        PolicyViolationError,
        match="No policy registered for tenant 'tenant-unregistered'",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=request,
        )

    repository.create.assert_not_called()
    repository.update.assert_not_called()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_persists_authenticated_tenant_in_run_and_snapshot() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    request = AgentRequest(
        input="Explain the platform",
        session_id="session-tenant-1",
        user_id="user-tenant-1",
        principal="api_key:tenant-principal",
        tenant_id="tenant-acme",
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=request,
    )

    assert result.run_id

    pending = repository.create.call_args.args[0]
    running = repository.running_run
    completed = repository.completed_run

    assert running is not None

    assert pending.tenant_id == "tenant-acme"
    assert pending.request_snapshot.tenant_id == "tenant-acme"

    assert running.tenant_id == "tenant-acme"
    assert running.request_snapshot.tenant_id == "tenant-acme"

    assert completed.tenant_id == "tenant-acme"
    assert completed.request_snapshot.tenant_id == "tenant-acme"


@pytest.mark.asyncio
async def test_execute_persists_pending_running_and_completed_lifecycle() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    response = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
        ),
    )

    assert isinstance(response, AgentRunExecutionResult)
    assert response.run_id
    assert response.response.output == "completed"

    assert repository.create.call_count == 1
    repository.claim_pending_run.assert_called_once()
    repository.complete_if_owner.assert_called_once()

    pending = repository.create.call_args.args[0]
    running = repository.running_run
    completed = repository.completed_run

    assert running is not None

    assert pending.status == AgentRunStatus.PENDING
    assert pending.agent_name == "enterprise-analyst"
    assert pending.session_id == "session-1"
    assert pending.user_id == "user-1"
    assert pending.started_at is None
    assert pending.completed_at is None

    assert running.run_id == pending.run_id
    assert running.lease_id is not None
    assert running.lease_expires_at is not None
    assert running.lease_expires_at > running.started_at

    assert repository.complete_if_owner.call_args.args[0] == running.run_id
    assert repository.complete_if_owner.call_args.kwargs["lease_id"] == running.lease_id
    assert running.status == AgentRunStatus.RUNNING
    assert running.started_at is not None
    assert running.completed_at is None

    assert completed.run_id == pending.run_id
    assert completed.status == AgentRunStatus.COMPLETED
    assert completed.started_at == running.started_at
    assert completed.completed_at is not None
    assert completed.output == "completed"
    assert completed.error_type is None
    assert completed.error_message is None
    assert completed.lease_id is None
    assert completed.lease_expires_at is None

    runtime.run.assert_awaited_once()

    runtime_call = runtime.run.await_args
    assert runtime_call.args == (
        "enterprise-analyst",
        AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            execution_budget=ExecutionBudget(),
        ),
    )
    assert runtime_call.kwargs["lease_id"] == running.lease_id
    assert runtime_call.kwargs["run_id"] == pending.run_id
    assert runtime_call.kwargs["execution_ownership_lost"] is not None
    assert runtime_call.kwargs["cancellation_requested"] is not None
    assert runtime_call.kwargs["cancellation_requested"].is_set() is False


@pytest.mark.asyncio
async def test_execute_persists_waiting_for_approval_lifecycle() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(
        side_effect=AgentExecutionWaitingForApprovalError(
            "Agent execution is waiting for human approval."
        ),
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentExecutionWaitingForApprovalError,
        match="waiting for human approval",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Execute the approved workflow",
                session_id="session-approval-1",
                user_id="user-approval-1",
            ),
        )

    assert repository.create.call_count == 1
    repository.claim_pending_run.assert_called_once()

    pending = repository.create.call_args.args[0]
    running = repository.running_run
    waiting = repository.waiting_for_approval_run

    assert running is not None
    assert waiting is not None
    assert pending.status == AgentRunStatus.PENDING
    assert running.status == AgentRunStatus.RUNNING
    assert waiting.run_id == pending.run_id
    assert waiting.status == AgentRunStatus.WAITING_FOR_APPROVAL
    assert waiting.started_at == running.started_at
    assert waiting.completed_at is None
    assert waiting.error_type is None
    assert waiting.error_message is None
    assert waiting.lease_id is None
    assert waiting.lease_expires_at is None

    repository.transition_to_waiting_for_approval_if_owner.assert_called_once()
    transition_call = repository.transition_to_waiting_for_approval_if_owner.call_args

    assert transition_call.args[0] == running.run_id
    assert transition_call.kwargs["lease_id"] == running.lease_id
    assert transition_call.kwargs["updated_at"] is not None

    repository.fail_if_owner.assert_not_called()
    repository.cancel_if_owner.assert_not_called()
    repository.complete_if_owner.assert_not_called()
    runtime.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_cancels_when_cancellation_is_requested_during_registration_gap():
    repository = _repository()

    class BlockingRuntime:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = False

        async def run(
            self,
            agent_name,
            request,
            *,
            lease_id=None,
            run_id=None,
            execution_ownership_lost=None,
            cancellation_requested=None,
        ):
            self.started.set()

            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    runtime = BlockingRuntime()

    running_run = None
    cancellation_checked = False

    original_claim = repository.claim_pending_run.side_effect

    def capture_running_run(
        run_id,
        *,
        started_at,
        lease_id,
        lease_expires_at,
    ):
        nonlocal running_run
        running_run = original_claim(
            run_id,
            started_at=started_at,
            lease_id=lease_id,
            lease_expires_at=lease_expires_at,
        )
        return running_run

    repository.claim_pending_run.side_effect = capture_running_run

    def get_with_registration_race(run_id: str):
        nonlocal cancellation_checked

        if running_run is None:
            return None

        if not cancellation_checked:
            cancellation_checked = True
            return running_run.model_copy(
                update={
                    "cancellation_requested": True,
                    "cancellation_requested_at": datetime.now(UTC),
                }
            )

        return running_run

    repository.get.side_effect = get_with_registration_race

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    execution_task = asyncio.create_task(
        service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
            ),
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    assert cancellation_checked is True
    assert runtime.cancelled is False
    repository.cancel_if_owner.assert_called_once()
    assert hasattr(repository, "cancelled_run")
    assert repository.cancelled_run.status is AgentRunStatus.CANCELLED
    repository.complete_if_owner.assert_not_called()
    repository.fail_if_owner.assert_not_called()


@pytest.mark.asyncio
async def test_execute_uses_one_run_id_across_lifecycle() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(input="Hello"),
    )

    created_run = repository.create.call_args.args[0]
    running_run = repository.running_run

    assert running_run is not None
    completed_run_id = repository.complete_if_owner.call_args.args[0]

    assert created_run.run_id == running_run.run_id
    assert completed_run_id == running_run.run_id


@pytest.mark.asyncio
async def test_execute_propagates_execution_ownership_loss_without_terminal_persistence():
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(
        side_effect=AgentExecutionOwnershipLostError("Agent execution lost durable run ownership."),
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentExecutionOwnershipLostError,
        match="lost durable run ownership",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Ownership loss"),
        )

    assert repository.create.call_count == 1
    repository.claim_pending_run.assert_called_once()
    repository.fail_if_owner.assert_not_called()
    repository.cancel_if_owner.assert_not_called()
    repository.complete_if_owner.assert_not_called()


@pytest.mark.asyncio
async def test_execute_preserves_agent_response_without_rebuilding_it() -> None:
    repository = _repository()

    response = _response(
        output={"answer": "structured"},
        session_id="session-42",
    )

    runtime = Mock()
    runtime.run = AsyncMock(return_value=response)

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Return structured output",
            session_id="session-42",
        ),
    )

    assert isinstance(result, AgentRunExecutionResult)
    assert result.response is response
    assert result.run_id


@pytest.mark.asyncio
async def test_execute_persists_failed_run_and_reraises_runtime_error() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(
        side_effect=RuntimeError("agent execution failed"),
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="agent execution failed"):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Fail"),
        )

    assert repository.create.call_count == 1
    repository.claim_pending_run.assert_called_once()
    repository.fail_if_owner.assert_called_once()

    pending = repository.create.call_args.args[0]
    running = repository.running_run
    failed = repository.failed_run

    assert running is not None

    assert pending.status == AgentRunStatus.PENDING
    assert running.status == AgentRunStatus.RUNNING
    assert running.lease_id is not None
    assert running.lease_expires_at is not None

    assert repository.fail_if_owner.call_args.args[0] == running.run_id
    assert repository.fail_if_owner.call_args.kwargs["lease_id"] == running.lease_id

    assert failed.run_id == pending.run_id
    assert failed.status == AgentRunStatus.FAILED
    assert failed.started_at == running.started_at
    assert failed.completed_at is not None
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "agent execution failed"
    assert failed.output is None
    assert failed.lease_id is None
    assert failed.lease_expires_at is None


@pytest.mark.asyncio
async def test_execute_persists_failed_run_for_lookup_error() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(
        side_effect=LookupError("agent not found"),
    )

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(LookupError, match="agent not found"):
        await service.execute(
            agent_name="missing-agent",
            request=AgentRequest(input="Hello"),
        )

    repository.fail_if_owner.assert_called_once()

    failed = repository.failed_run
    running = repository.running_run

    assert running is not None

    assert failed.status == AgentRunStatus.FAILED
    assert failed.error_type == "LookupError"
    assert failed.error_message == "agent not found"
    assert repository.fail_if_owner.call_args.args[0] == running.run_id
    assert repository.fail_if_owner.call_args.kwargs["lease_id"] == running.lease_id


@pytest.mark.asyncio
async def test_execute_does_not_mark_failed_when_running_persistence_fails() -> None:
    repository = _repository()
    repository.claim_pending_run.side_effect = RuntimeError("running persistence failed")

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="running persistence failed"):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Hello"),
        )

    runtime.run.assert_not_awaited()
    repository.create.assert_called_once()
    repository.claim_pending_run.assert_called_once()


@pytest.mark.asyncio
async def test_execute_does_not_attempt_failed_update_when_runtime_never_starts() -> None:
    repository = Mock(spec=AgentRunRepository)
    repository.create.side_effect = RuntimeError("initial persistence failed")

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="initial persistence failed"):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Hello"),
        )

    repository.update.assert_not_called()
    runtime.run.assert_not_awaited()


def test_get_run_returns_repository_result() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
        output="completed",
    )
    repository.get.return_value = run

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = service.get_run("run-123")

    assert result is run
    repository.get.assert_called_once_with("run-123")


@pytest.mark.asyncio
async def test_execute_preserves_runtime_error_when_failed_persistence_fails() -> None:
    repository = _repository()

    original_error = RuntimeError("agent execution failed")
    runtime = Mock()
    runtime.run = AsyncMock(side_effect=original_error)

    repository.fail_if_owner.side_effect = RuntimeError("failed-state persistence failed")

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="agent execution failed") as exc_info:
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Fail"),
        )

    assert exc_info.value is original_error
    assert repository.create.call_count == 1
    repository.claim_pending_run.assert_called_once()
    repository.fail_if_owner.assert_called_once()


@pytest.mark.asyncio
async def test_execute_surfaces_completed_persistence_failure() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    repository.complete_if_owner.side_effect = RuntimeError("completed-state persistence failed")

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="completed-state persistence failed",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(input="Complete"),
        )

    runtime.run.assert_awaited_once()
    assert repository.create.call_count == 1
    repository.claim_pending_run.assert_called_once()
    repository.complete_if_owner.assert_called_once()


def test_get_run_returns_none_for_missing_run() -> None:
    repository = _repository()
    repository.get.return_value = None

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = service.get_run("missing-run")

    assert result is None
    repository.get.assert_called_once_with("missing-run")


def test_get_run_rejects_partial_identity_context() -> None:
    repository = _repository()
    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    with pytest.raises(
        AgentRunAccessDeniedError,
        match="Both tenant_id and principal are required",
    ):
        service.get_run(
            "run-123",
            tenant_id="tenant-acme",
        )

    repository.get.assert_not_called()
    repository.get_for_tenant.assert_not_called()


def test_cancel_rejects_partial_identity_context() -> None:
    repository = _repository()
    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    with pytest.raises(
        AgentRunAccessDeniedError,
        match="Both tenant_id and principal are required",
    ):
        service.cancel(
            "run-123",
            principal="api_key:test-owner",
        )

    repository.get.assert_not_called()
    repository.get_for_tenant.assert_not_called()


def test_list_events_rejects_partial_identity_context() -> None:
    repository = _repository()
    events_repository = Mock(spec=AgentRunEventsRepository)

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        events_repository=events_repository,
    )

    with pytest.raises(
        AgentRunAccessDeniedError,
        match="Both tenant_id and principal are required",
    ):
        service.list_events(
            "run-123",
            tenant_id="tenant-acme",
        )

    repository.get.assert_not_called()
    repository.get_for_tenant.assert_not_called()
    events_repository.list.assert_not_called()


def test_list_runs_delegates_filters_and_limit() -> None:
    repository = _repository()

    runs = [
        AgentRun(
            run_id="run-1",
            agent_name="enterprise-analyst",
            status=AgentRunStatus.COMPLETED,
        ),
        AgentRun(
            run_id="run-2",
            agent_name="enterprise-analyst",
            status=AgentRunStatus.FAILED,
        ),
    ]
    repository.list.return_value = runs

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = service.list_runs(
        agent_name="enterprise-analyst",
        session_id="session-1",
        user_id="user-1",
        status=AgentRunStatus.COMPLETED,
        limit=25,
    )

    assert result == runs
    repository.list.assert_called_once_with(
        tenant_id=None,
        agent_name="enterprise-analyst",
        session_id="session-1",
        user_id="user-1",
        status=AgentRunStatus.COMPLETED,
        limit=25,
    )


def test_list_events_delegates_to_event_repository() -> None:
    repository = _repository()
    events_repository = Mock(spec=AgentRunEventsRepository)

    run = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )
    repository.get.return_value = run

    events = [
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_STARTED,
            agent_name="enterprise-analyst",
            run_id="run-123",
        ),
        AgentExecutionEvent(
            event_type=AgentExecutionEventType.AGENT_COMPLETED,
            agent_name="enterprise-analyst",
            run_id="run-123",
        ),
    ]
    events_repository.list.return_value = events

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        events_repository=events_repository,
    )

    result = service.list_events(
        "run-123",
        limit=25,
    )

    assert result == events
    repository.get.assert_called_once_with("run-123")
    events_repository.list.assert_called_once_with(
        "run-123",
        limit=25,
    )


def test_list_events_raises_for_missing_run() -> None:
    repository = _repository()
    repository.get.return_value = None

    events_repository = Mock(spec=AgentRunEventsRepository)
    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        events_repository=events_repository,
    )

    with pytest.raises(
        LookupError,
        match="Agent run 'missing-run' was not found.",
    ):
        service.list_events("missing-run")

    repository.get.assert_called_once_with("missing-run")
    events_repository.list.assert_not_called()


def test_list_events_requires_event_repository() -> None:
    repository = _repository()
    repository.get.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )

    runtime = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent run events repository is not configured.",
    ):
        service.list_events("run-123")


@pytest.mark.asyncio
async def test_execute_rejects_run_before_runtime_when_admission_denied() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    admission_policy = Mock()
    admission_policy.evaluate = AsyncMock(
        return_value=AgentRunAdmissionResult(
            allowed=False,
            reason="agent is not approved for this environment",
        )
    )

    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        admission_policy=admission_policy,
        observer=observer,
    )

    with pytest.raises(
        AgentRunAdmissionRejectedError,
        match="agent is not approved for this environment",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
            ),
        )

    assert repository.create.call_count == 1
    repository.claim_pending_run.assert_not_called()
    runtime.run.assert_not_awaited()

    pending = repository.create.call_args.args[0]
    rejected = repository.update.call_args.args[0]

    assert pending.status == AgentRunStatus.PENDING
    assert pending.started_at is None
    assert pending.completed_at is None

    assert rejected.run_id == pending.run_id
    assert rejected.status == AgentRunStatus.REJECTED
    assert rejected.started_at is None
    assert rejected.completed_at is not None
    assert rejected.metadata["admission"] == {
        "allowed": False,
        "reason": "agent is not approved for this environment",
    }

    assert len(observer.events) == 1
    governance_event = observer.events[0]
    assert governance_event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert governance_event.run_id == pending.run_id
    assert governance_event.agent_name == "enterprise-analyst"
    assert governance_event.metadata == {
        "governance_domain": "admission",
        "decision": "deny",
        "reason": "agent is not approved for this environment",
    }

    admission_policy.evaluate.assert_awaited_once()
    call = admission_policy.evaluate.await_args

    assert call.kwargs["agent_name"] == "enterprise-analyst"
    assert call.kwargs["request"].user_id == "user-1"
    assert call.kwargs["run"].run_id == pending.run_id
    assert call.kwargs["run"].status == AgentRunStatus.PENDING


@pytest.mark.asyncio
async def test_execute_allows_run_when_admission_policy_allows() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    admission_policy = Mock()
    admission_policy.evaluate = AsyncMock(
        return_value=AgentRunAdmissionResult(
            allowed=True,
        )
    )

    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        admission_policy=admission_policy,
        observer=observer,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(input="Explain the platform"),
    )

    assert result.response.output == "completed"
    runtime.run.assert_awaited_once()

    assert len(observer.events) == 1
    governance_event = observer.events[0]
    assert governance_event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert governance_event.run_id is not None
    assert governance_event.agent_name == "enterprise-analyst"
    assert governance_event.metadata == {
        "governance_domain": "admission",
        "decision": "allow",
    }

    assert repository.create.call_count == 1
    repository.claim_pending_run.assert_called_once()
    repository.complete_if_owner.assert_called_once()

    running = repository.running_run
    completed = repository.completed_run

    assert running is not None

    assert running.status == AgentRunStatus.RUNNING
    assert running.lease_id is not None
    assert running.lease_expires_at is not None
    assert running.lease_expires_at > running.started_at

    assert repository.complete_if_owner.call_args.args[0] == running.run_id
    assert repository.complete_if_owner.call_args.kwargs["lease_id"] == running.lease_id

    assert completed.status == AgentRunStatus.COMPLETED
    assert completed.lease_id is None
    assert completed.lease_expires_at is None


@pytest.mark.asyncio
async def test_heartbeat_loop_renews_owned_running_run() -> None:
    repository = Mock(spec=AgentRunRepository)

    renewed_run = AgentRun(
        run_id="run-heartbeat",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    repository.heartbeat.return_value = renewed_run

    sleep_calls = 0

    async def sleep_once(seconds: float) -> None:
        nonlocal sleep_calls

        assert seconds == 20
        sleep_calls += 1

        if sleep_calls == 2:
            raise asyncio.CancelledError

    with patch(
        "app.control_plane.agent_runs.lease.asyncio.sleep",
        side_effect=sleep_once,
    ):
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_loop(
                repository,
                run_id="run-heartbeat",
                lease_id="lease-123",
                lease_seconds=60,
            )

    repository.heartbeat.assert_called_once()

    call = repository.heartbeat.call_args
    assert call.args[0] == "run-heartbeat"
    assert call.kwargs["lease_id"] == "lease-123"
    assert call.kwargs["lease_expires_at"] > datetime.now(UTC)

    assert sleep_calls == 2


@pytest.mark.asyncio
async def test_heartbeat_loop_stops_when_lease_ownership_is_lost() -> None:
    repository = Mock(spec=AgentRunRepository)
    repository.heartbeat.return_value = None

    async def sleep_once(seconds: float) -> None:
        assert seconds == 20

    with patch(
        "app.control_plane.agent_runs.lease.asyncio.sleep",
        side_effect=sleep_once,
    ):
        await heartbeat_loop(
            repository,
            run_id="run-heartbeat",
            lease_id="lease-123",
            lease_seconds=60,
        )

    repository.heartbeat.assert_called_once()


@pytest.mark.asyncio
async def test_heartbeat_loop_survives_heartbeat_persistence_failure() -> None:
    repository = Mock(spec=AgentRunRepository)

    renewed_run = AgentRun(
        run_id="run-heartbeat",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
    )
    repository.heartbeat.side_effect = [RuntimeError("database unavailable"), renewed_run]

    sleep_calls = 0

    async def sleep_once(seconds: float) -> None:
        nonlocal sleep_calls

        assert seconds == 20
        sleep_calls += 1

        if sleep_calls == 3:
            raise asyncio.CancelledError

    with patch(
        "app.control_plane.agent_runs.lease.asyncio.sleep",
        side_effect=sleep_once,
    ):
        with pytest.raises(asyncio.CancelledError):
            await heartbeat_loop(
                repository,
                run_id="run-heartbeat",
                lease_id="lease-123",
                lease_seconds=60,
            )

    assert repository.heartbeat.call_count == 2
    assert sleep_calls == 3


@pytest.mark.asyncio
async def test_execute_starts_and_cancels_heartbeat_task() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        lease_seconds=60,
    )

    original_create_task = asyncio.create_task
    created_tasks = []

    def create_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    with patch(
        "app.control_plane.agent_runs.application_service.asyncio.create_task",
        side_effect=create_task,
    ) as create_task_mock:
        result = await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
            ),
        )

    assert result.response.output == "completed"
    create_task_mock.assert_called_once()

    assert len(created_tasks) == 1
    assert created_tasks[0].done()
    assert created_tasks[0].cancelled()

    repository.complete_if_owner.assert_called_once()


@pytest.mark.asyncio
async def test_execute_persists_cancelled_run_and_emits_audit_event() -> None:
    repository = _repository()

    runtime = Mock()

    started = asyncio.Event()

    async def run_agent(*args, **kwargs):
        started.set()

        cancellation_requested = kwargs["cancellation_requested"]
        await cancellation_requested.wait()

        raise asyncio.CancelledError("Agent run cancellation requested.")

    runtime.run = AsyncMock(side_effect=run_agent)

    from app.control_plane.agent_runs.cancellation import (
        AgentRunCancellationRegistry,
    )

    registry = AgentRunCancellationRegistry()
    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        observer=observer,
        cancellation_registry=registry,
    )

    execution_task = asyncio.create_task(
        service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Long running task",
                session_id="session-1",
                user_id="user-1",
            ),
        )
    )

    await started.wait()

    running = repository.running_run

    assert running is not None
    assert running.status == AgentRunStatus.RUNNING
    assert registry.cancel(running.run_id) is True

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    repository.cancel_if_owner.assert_called_once()

    runtime_call = runtime.run.await_args
    cancellation_requested = runtime_call.kwargs["cancellation_requested"]
    assert cancellation_requested.is_set() is True

    cancelled = repository.cancelled_run

    assert cancelled.run_id == running.run_id
    assert cancelled.status == AgentRunStatus.CANCELLED
    assert cancelled.started_at == running.started_at
    assert cancelled.completed_at is not None
    assert cancelled.output is None
    assert cancelled.error_type is None
    assert cancelled.error_message is None
    assert cancelled.lease_id is None
    assert cancelled.lease_expires_at is None

    assert repository.cancel_if_owner.call_args.args[0] == running.run_id
    assert repository.cancel_if_owner.call_args.kwargs["lease_id"] == running.lease_id
    assert repository.cancel_if_owner.call_args.kwargs["completed_at"] is not None

    repository.fail_if_owner.assert_not_called()
    repository.complete_if_owner.assert_not_called()

    assert len(observer.events) == 2

    governance_event = next(
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    )
    assert governance_event.run_id == running.run_id
    assert governance_event.agent_name == running.agent_name
    assert governance_event.metadata == {
        "governance_domain": "admission",
        "decision": "allow",
    }

    event = next(
        event
        for event in observer.events
        if event.event_type is AgentExecutionEventType.AGENT_CANCELLED
    )

    assert event.agent_name == running.agent_name
    assert event.run_id == running.run_id
    assert event.session_id == running.session_id
    assert event.user_id == running.user_id
    assert event.metadata == {}
    assert event.tool_round is None
    assert event.tool_name is None
    assert event.call_id is None
    assert event.provider is None
    assert event.model is None


@pytest.mark.asyncio
async def test_execute_cancellation_does_not_emit_event_after_ownership_loss() -> None:
    repository = _repository()
    repository.cancel_if_owner.side_effect = lambda *args, **kwargs: None

    runtime = Mock()

    started = asyncio.Event()

    async def run_agent(*args, **kwargs):
        started.set()

        cancellation_requested = kwargs["cancellation_requested"]
        await cancellation_requested.wait()

        raise asyncio.CancelledError("Agent run cancellation requested.")

    runtime.run = AsyncMock(side_effect=run_agent)

    from app.control_plane.agent_runs.cancellation import (
        AgentRunCancellationRegistry,
    )

    registry = AgentRunCancellationRegistry()
    observer = RecordingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        observer=observer,
        cancellation_registry=registry,
    )

    execution_task = asyncio.create_task(
        service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Long running task",
                session_id="session-1",
                user_id="user-1",
            ),
        )
    )

    await started.wait()

    running = repository.running_run
    assert running is not None

    assert registry.cancel(running.run_id) is True

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    repository.cancel_if_owner.assert_called_once()

    assert len(observer.events) == 1
    governance_event = observer.events[0]
    assert governance_event.event_type is AgentExecutionEventType.GOVERNANCE_DECISION
    assert governance_event.run_id == running.run_id
    assert governance_event.agent_name == running.agent_name
    assert governance_event.metadata == {
        "governance_domain": "admission",
        "decision": "allow",
    }


@pytest.mark.asyncio
async def test_execute_cancellation_continues_when_observer_fails() -> None:
    repository = _repository()

    runtime = Mock()

    started = asyncio.Event()

    async def run_agent(*args, **kwargs):
        started.set()

        cancellation_requested = kwargs["cancellation_requested"]
        await cancellation_requested.wait()

        raise asyncio.CancelledError("Agent run cancellation requested.")

    runtime.run = AsyncMock(side_effect=run_agent)

    from app.control_plane.agent_runs.cancellation import (
        AgentRunCancellationRegistry,
    )

    registry = AgentRunCancellationRegistry()
    observer = FailingObserver()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        observer=observer,
        cancellation_registry=registry,
    )

    execution_task = asyncio.create_task(
        service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Long running task",
                session_id="session-1",
                user_id="user-1",
            ),
        )
    )

    await started.wait()

    running = repository.running_run
    assert running is not None

    assert registry.cancel(running.run_id) is True

    with pytest.raises(asyncio.CancelledError):
        await execution_task

    repository.cancel_if_owner.assert_called_once()

    cancelled = repository.cancelled_run
    assert cancelled.status == AgentRunStatus.CANCELLED
    assert observer.calls == 2


@pytest.mark.asyncio
async def test_cancellation_registry_sets_signal_without_cancelling_task():
    from app.control_plane.agent_runs.cancellation import (
        AgentRunCancellationRegistry,
    )

    registry = AgentRunCancellationRegistry()
    cancellation_requested = asyncio.Event()

    async def worker():
        await asyncio.Event().wait()

    task = asyncio.create_task(worker())

    registry.register(
        "run-123",
        task,
        cancellation_requested,
    )

    assert registry.cancel("run-123") is True
    assert cancellation_requested.is_set() is True
    assert task.cancelled() is False
    assert task.done() is False

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


def test_cancel_raises_lookup_error_for_missing_run() -> None:
    repository = _repository()
    repository.get.return_value = None

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        LookupError,
        match="Agent run 'missing-run' was not found.",
    ):
        service.cancel("missing-run")

    repository.get.assert_called_once_with("missing-run")


def test_cancel_rejects_non_running_run() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="completed-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.COMPLETED,
    )
    repository.get.return_value = run

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        ValueError,
        match="Agent run 'completed-run' is not cancellable from status 'completed'",
    ):
        service.cancel("completed-run")

    repository.get.assert_called_once_with("completed-run")
    repository.request_cancellation.assert_not_called()


def test_cancel_persists_intent_then_signals_registered_running_task() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="running-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        started_at=datetime.now(UTC),
    )
    requested_run = run.model_copy(
        update={
            "cancellation_requested": True,
            "cancellation_requested_at": datetime.now(UTC),
        }
    )
    repository.get.return_value = run
    repository.request_cancellation.return_value = requested_run

    cancellation_registry = Mock()
    cancellation_registry.cancel.return_value = True

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        cancellation_registry=cancellation_registry,
    )

    result = service.cancel("running-run")

    assert result is requested_run
    repository.get.assert_called_once_with("running-run")
    repository.request_cancellation.assert_called_once()
    request_kwargs = repository.request_cancellation.call_args.kwargs
    assert request_kwargs["requested_at"].tzinfo is UTC
    cancellation_registry.cancel.assert_called_once_with("running-run")


def test_cancel_returns_durable_intent_when_local_task_is_missing() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="running-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        started_at=datetime.now(UTC),
    )
    requested_run = run.model_copy(
        update={
            "cancellation_requested": True,
            "cancellation_requested_at": datetime.now(UTC),
        }
    )
    repository.get.return_value = run
    repository.request_cancellation.return_value = requested_run

    cancellation_registry = Mock()
    cancellation_registry.cancel.return_value = False

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        cancellation_registry=cancellation_registry,
    )

    result = service.cancel("running-run")

    assert result is requested_run
    repository.request_cancellation.assert_called_once()
    cancellation_registry.cancel.assert_called_once_with("running-run")


def test_cancel_rejects_when_cancellation_intent_cannot_be_persisted() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="running-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        started_at=datetime.now(UTC),
    )
    repository.get.return_value = run
    repository.request_cancellation.return_value = None

    runtime = Mock()
    cancellation_registry = Mock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
        cancellation_registry=cancellation_registry,
    )

    with pytest.raises(
        RuntimeError,
        match="could not accept cancellation",
    ):
        service.cancel("running-run")

    repository.request_cancellation.assert_called_once()
    cancellation_registry.cancel.assert_not_called()


def test_cancel_does_not_require_local_registry_when_intent_is_persisted() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="running-run",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        lease_id="lease-123",
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=60),
        started_at=datetime.now(UTC),
    )
    requested_run = run.model_copy(
        update={
            "cancellation_requested": True,
            "cancellation_requested_at": datetime.now(UTC),
        }
    )
    repository.get.return_value = run
    repository.request_cancellation.return_value = requested_run

    runtime = Mock()
    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = service.cancel("running-run")

    assert result is requested_run
    repository.request_cancellation.assert_called_once()


def _idempotent_run(
    *,
    status: AgentRunStatus = AgentRunStatus.COMPLETED,
    run_id: str = "existing-run",
    agent_name: str = "enterprise-analyst",
    output="completed",
    user_id: str = "user-1",
    idempotency_key: str = "request-123",
) -> AgentRun:
    request = AgentRequest(
        input="Explain the platform",
        session_id="session-1",
        user_id=user_id,
        execution_budget=ExecutionBudget(),
        metadata={"request": "same"},
    )

    return AgentRun(
        run_id=run_id,
        agent_name=agent_name,
        session_id=request.session_id,
        user_id=user_id,
        idempotency_key=idempotency_key,
        status=status,
        output=output,
        metadata={"source": "persisted"},
        request_snapshot=AgentRunRequestSnapshot.from_request(request),
        error_message=("persisted failure" if status is AgentRunStatus.FAILED else None),
    )


@pytest.mark.asyncio
async def test_execute_without_idempotency_key_creates_new_run() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            user_id="user-1",
        ),
    )

    repository.get_by_idempotency_key.assert_not_called()
    repository.create.assert_called_once()
    runtime.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_with_new_idempotency_key_creates_run() -> None:
    repository = _repository()
    repository.get_by_idempotency_key.return_value = None

    runtime = Mock()
    runtime.run = AsyncMock(return_value=_response())

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            user_id="user-1",
        ),
        idempotency_key="request-123",
    )

    repository.get_by_idempotency_key.assert_called_once_with(
        None,
        "user-1",
        "request-123",
    )

    created = repository.create.call_args.args[0]
    assert created.idempotency_key == "request-123"
    runtime.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_execute_replays_completed_idempotent_run() -> None:
    repository = _repository()

    existing = _idempotent_run(
        output={"answer": "persisted"},
    )
    repository.get_by_idempotency_key.return_value = existing

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            metadata={"request": "same"},
        ),
        idempotency_key="request-123",
    )

    assert result.run_id == "existing-run"
    assert result.response == AgentResponse(
        agent_name="enterprise-analyst",
        output={"answer": "persisted"},
        session_id="session-1",
        metadata={"source": "persisted"},
    )
    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_reuses_running_idempotent_run_reference() -> None:
    repository = _repository()

    existing = _idempotent_run(
        status=AgentRunStatus.RUNNING,
        output=None,
    )
    repository.get_by_idempotency_key.return_value = existing

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            metadata={"request": "same"},
        ),
        idempotency_key="request-123",
    )

    assert result.run_id == "existing-run"
    assert result.response.agent_name == "enterprise-analyst"
    assert result.response.output is None
    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_reuses_pending_idempotent_run_reference() -> None:
    repository = _repository()

    existing = _idempotent_run(
        status=AgentRunStatus.PENDING,
        output=None,
    )
    repository.get_by_idempotency_key.return_value = existing

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            metadata={"request": "same"},
        ),
        idempotency_key="request-123",
    )

    assert result.run_id == "existing-run"
    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_rejects_idempotency_key_for_different_request() -> None:
    repository = _repository()

    repository.get_by_idempotency_key.return_value = _idempotent_run()

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentRunIdempotencyConflictError,
        match="different request",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="A different question",
                session_id="session-1",
                user_id="user-1",
                metadata={"request": "same"},
            ),
            idempotency_key="request-123",
        )

    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_rejects_idempotency_key_for_different_agent() -> None:
    repository = _repository()

    repository.get_by_idempotency_key.return_value = _idempotent_run(
        agent_name="different-agent",
    )

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentRunIdempotencyConflictError,
        match="different agent",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
                metadata={"request": "same"},
            ),
            idempotency_key="request-123",
        )

    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_surfaces_previous_failed_idempotent_run() -> None:
    repository = _repository()

    repository.get_by_idempotency_key.return_value = _idempotent_run(
        status=AgentRunStatus.FAILED,
    )

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="previously failed: persisted failure",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="session-1",
                user_id="user-1",
                metadata={"request": "same"},
            ),
            idempotency_key="request-123",
        )

    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


@pytest.mark.asyncio
async def test_execute_requires_user_for_idempotency_key() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        ValueError,
        match="user_id is required",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
            ),
            idempotency_key="request-123",
        )

    repository.get_by_idempotency_key.assert_not_called()
    repository.create.assert_not_called()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_rejects_blank_idempotency_key() -> None:
    repository = _repository()

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        ValueError,
        match="Idempotency key must not be empty",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                user_id="user-1",
            ),
            idempotency_key="   ",
        )

    repository.get_by_idempotency_key.assert_not_called()
    repository.create.assert_not_called()
    runtime.run.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_race_on_idempotent_create_reloads_existing_run() -> None:
    repository = _repository()

    existing = _idempotent_run()

    repository.get_by_idempotency_key.side_effect = [
        None,
        existing,
    ]
    repository.create.side_effect = DuplicateAgentRunError("agent run already exists")

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    result = await service.execute(
        agent_name="enterprise-analyst",
        request=AgentRequest(
            input="Explain the platform",
            session_id="session-1",
            user_id="user-1",
            metadata={"request": "same"},
        ),
        idempotency_key="request-123",
    )

    assert result.run_id == "existing-run"
    runtime.run.assert_not_awaited()
    assert repository.get_by_idempotency_key.call_count == 2
    repository.create.assert_called_once()


@pytest.mark.asyncio
async def test_execute_rejects_idempotency_key_for_different_session() -> None:
    repository = _repository()

    repository.get_by_idempotency_key.return_value = _idempotent_run(
        output="persisted",
    )

    runtime = Mock()
    runtime.run = AsyncMock()

    service = AgentRunApplicationService(
        runtime=runtime,
        repository=repository,
    )

    with pytest.raises(
        AgentRunIdempotencyConflictError,
        match="different request",
    ):
        await service.execute(
            agent_name="enterprise-analyst",
            request=AgentRequest(
                input="Explain the platform",
                session_id="different-session",
                user_id="user-1",
                metadata={"request": "same"},
            ),
            idempotency_key="request-123",
        )

    runtime.run.assert_not_awaited()
    repository.create.assert_not_called()


def _step(
    *,
    run_id: str = "run-123",
    step_id: str = "step-1",
    step_index: int = 0,
    status: AgentRunStepStatus = AgentRunStepStatus.COMPLETED,
) -> AgentRunStep:
    return AgentRunStep(
        run_id=run_id,
        step_id=step_id,
        step_index=step_index,
        step_type="tool_execution",
        status=status,
        attempt=1,
        tool_name="vehicle_query",
        call_id="call-1",
        input={"query": "vehicle events"},
        output={"rows": 3},
        metadata={"source": "test"},
    )


def test_list_steps_delegates_to_step_repository() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    steps_repository = Mock(spec=AgentRunStepsRepository)
    steps = [
        _step(step_id="step-1", step_index=0),
        _step(step_id="step-2", step_index=1),
    ]
    steps_repository.list.return_value = steps

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    result = service.list_steps(
        "run-123",
        tenant_id="tenant-acme",
        principal="api_key:test-owner",
        status=AgentRunStepStatus.COMPLETED,
        limit=25,
    )

    assert result == steps
    repository.get_for_tenant.assert_called_once_with(
        "run-123",
        "tenant-acme",
    )
    steps_repository.list.assert_called_once_with(
        "run-123",
        status=AgentRunStepStatus.COMPLETED,
        limit=25,
    )


def test_list_steps_raises_for_missing_run() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = None

    steps_repository = Mock(spec=AgentRunStepsRepository)

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    with pytest.raises(
        LookupError,
        match="Agent run 'missing-run' was not found.",
    ):
        service.list_steps(
            "missing-run",
            tenant_id="tenant-acme",
            principal="api_key:test-owner",
        )

    repository.get_for_tenant.assert_called_once_with(
        "missing-run",
        "tenant-acme",
    )
    steps_repository.list.assert_not_called()


def test_list_steps_requires_step_repository() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent run steps repository is not configured.",
    ):
        service.list_steps(
            "run-123",
            tenant_id="tenant-acme",
            principal="api_key:test-owner",
        )


def test_get_step_delegates_to_step_repository() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    steps_repository = Mock(spec=AgentRunStepsRepository)
    step = _step()
    steps_repository.get.return_value = step

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    result = service.get_step(
        "run-123",
        "step-1",
        tenant_id="tenant-acme",
        principal="api_key:test-owner",
    )

    assert result is step
    repository.get_for_tenant.assert_called_once_with(
        "run-123",
        "tenant-acme",
    )
    steps_repository.get.assert_called_once_with(
        "run-123",
        "step-1",
    )


def test_get_step_returns_none_for_missing_step() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    steps_repository = Mock(spec=AgentRunStepsRepository)
    steps_repository.get.return_value = None

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    result = service.get_step(
        "run-123",
        "missing-step",
        tenant_id="tenant-acme",
        principal="api_key:test-owner",
    )

    assert result is None
    steps_repository.get.assert_called_once_with(
        "run-123",
        "missing-step",
    )


def test_get_step_raises_for_missing_run() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = None

    steps_repository = Mock(spec=AgentRunStepsRepository)

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
        agent_run_steps_repository=steps_repository,
    )

    with pytest.raises(
        LookupError,
        match="Agent run 'missing-run' was not found.",
    ):
        service.get_step(
            "missing-run",
            "step-1",
            tenant_id="tenant-acme",
            principal="api_key:test-owner",
        )

    repository.get_for_tenant.assert_called_once_with(
        "missing-run",
        "tenant-acme",
    )
    steps_repository.get.assert_not_called()


def test_get_step_requires_step_repository() -> None:
    repository = _repository()
    repository.get_for_tenant.return_value = AgentRun(
        run_id="run-123",
        agent_name="enterprise-analyst",
        principal="api_key:test-owner",
        tenant_id="tenant-acme",
        status=AgentRunStatus.COMPLETED,
    )

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    with pytest.raises(
        RuntimeError,
        match="Agent run steps repository is not configured.",
    ):
        service.get_step(
            "run-123",
            "step-1",
            tenant_id="tenant-acme",
            principal="api_key:test-owner",
        )


@pytest.mark.asyncio
async def test_execute_existing_run_raises_when_already_executing() -> None:
    repository = _repository()

    run = AgentRun(
        run_id="run-already-running",
        agent_name="enterprise-analyst",
        status=AgentRunStatus.RUNNING,
        user_id="user-1",
        principal="principal-1",
        tenant_id="tenant-1",
        session_id="session-1",
    )
    repository.running_run = run

    service = AgentRunApplicationService(
        runtime=Mock(),
        repository=repository,
    )

    request = AgentRequest(
        input="test request",
        session_id="session-1",
        user_id="user-1",
        principal="principal-1",
        tenant_id="tenant-1",
    )

    with pytest.raises(
        AgentRunAlreadyExecutingError,
        match="is already executing",
    ):
        await service.execute_existing_run(
            run_id=run.run_id,
            agent_name=run.agent_name,
            request=request,
        )

    repository.claim_pending_run.assert_not_called()
