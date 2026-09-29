"""End-to-end PostgreSQL integration test for approval-gated tool execution."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import delete, select

from ai_platform.agents.checkpoint import (
    AgentCheckpointPosition,
    AgentExecutionCheckpoint,
)
from ai_platform.agents.llm_agent import LLMAgent
from ai_platform.agents.llm_messages import assistant_tool_call_message, user_message
from ai_platform.agents.models import AgentRequest
from ai_platform.agents.registry import InMemoryAgentRegistry
from ai_platform.agents.runtime import AgentRuntime
from ai_platform.agents.tool_calls import AgentToolCall
from app.control_plane.agent_checkpoints.postgres_handler import (
    PostgreSQLAgentCheckpointHandler,
)
from app.control_plane.agent_checkpoints.postgres_repository import (
    PostgreSQLAgentCheckpointsRepository,
)
from ai_platform.agents.execution import AgentExecutionContext
from ai_platform.agents.orchestration import (
    OrchestrationPlan,
    OrchestrationStep,
    OrchestrationStepCompletionPolicy,
    OrchestrationStepStatus,
)
from app.control_plane.agent_run_events.postgres_observer import (
    PostgreSQLAgentRunEventObserver,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_steps.postgres_repository import (
    PostgreSQLAgentRunStepsRepository,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.postgres_repository import (
    PostgreSQLAgentRunRepository,
)
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.agent_runs.recovery_service import AgentRunRecoveryService
from app.control_plane.approvals.continuation_service import (
    AgentRunApprovalContinuationService,
)
from app.control_plane.approvals.coordinator import ControlPlaneApprovalCoordinator
from app.control_plane.approvals.models import ApprovalStatus
from app.control_plane.approvals.policy import SideEffectApprovalPolicy
from app.control_plane.approvals.postgres_repository import (
    PostgreSQLApprovalRequestRepository,
)
from app.control_plane.persistence.database import SessionLocal
from app.control_plane.persistence.models import (
    AgentRunCheckpointRecord,
    AgentRunEventRecord,
    AgentRunRecord,
    AgentRunStepRecord,
    ApprovalRequestRecord,
    ToolExecutionIdempotencyRecord,
)
from app.control_plane.tool_execution.postgres_idempotency import (
    PostgreSQLToolExecutionIdempotencyStore,
)
from tools.authorization.in_memory import InMemoryToolAuthorizer
from tools.authorization.service import ToolAuthorizationService
from tools.execution.exceptions import ToolExecutionWaitingForApprovalError
from tools.execution.context import ToolExecutionContext
from tools.registry.in_memory import InMemoryToolRegistry
from tools.execution.service import ToolExecutionService
from tools.models import ToolDefinition

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="Set RUN_POSTGRES_INTEGRATION=1 to run the PostgreSQL integration test",
)


class SideEffectTool:
    """Deterministic side-effecting tool whose invocation count is observable."""

    def __init__(self, name: str = "transfer_funds") -> None:
        self._definition = ToolDefinition(
            name=name,
            description="Transfers funds as a side effect.",
            metadata={
                "side_effect": True,
                "risk_tier": "high",
                "capability": "financial_transfer",
                "permission_scope": "payments.write",
            },
        )
        self.execution_count = 0
        self.executed_arguments: list[dict[str, object]] = []

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments):
        self.execution_count += 1
        self.executed_arguments.append(dict(arguments))
        return {
            "status": "transferred",
            "amount": arguments["amount"],
            "recipient": arguments["recipient"],
        }


class ApprovalContinuationPlanProvider:
    def build_plan(
        self,
        context: AgentExecutionContext,
    ) -> OrchestrationPlan:
        return OrchestrationPlan(
            steps=(
                OrchestrationStep(
                    step_id="retrieve_evidence",
                    step_index=0,
                    name="Execute approved side-effect tool",
                    status=OrchestrationStepStatus.PENDING,
                    completion_policy=OrchestrationStepCompletionPolicy.ON_TOOL_RESULT,
                    metadata={
                        "phase": "approval_continuation",
                        "completion_tool_name": "transfer_funds",
                    },
                ),
                OrchestrationStep(
                    step_id="produce_answer",
                    step_index=1,
                    name="Produce continuation response",
                    status=OrchestrationStepStatus.PENDING,
                    completion_policy=OrchestrationStepCompletionPolicy.ON_AGENT_RESPONSE,
                    metadata={
                        "phase": "response_generation",
                    },
                ),
            )
        )


class ContinuationLLMGateway:
    """Deterministic final-answer gateway for approval continuation."""

    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    async def route_chat(self, request):
        self.calls += 1
        self.requests.append(request)
        return {
            "provider": "integration-test",
            "model": "mock-gpt",
            "reply": "Transfer completed successfully.",
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "tool_calls": [],
        }


class CrashAfterApprovalRuntime:
    """Runtime that fails after approval has been durably committed."""

    async def resume(
        self,
        agent_name,
        request,
        checkpoint,
        *,
        run_id=None,
        lease_id=None,
        execution_ownership_lost=None,
    ):
        raise RuntimeError("simulated approval continuation crash")


def _build_approval_coordinator() -> ControlPlaneApprovalCoordinator:
    return ControlPlaneApprovalCoordinator(
        policy=SideEffectApprovalPolicy(
            approval_risk_tiers={"high"},
        ),
        repository_factory=lambda: PostgreSQLApprovalRequestRepository(SessionLocal()),
    )


def _build_runtime(
    *,
    execution_service: ToolExecutionService,
    agent_definition,
    llm_gateway: ContinuationLLMGateway,
    agent_run_steps_repository_factory,
) -> AgentRuntime:
    agent_registry = InMemoryAgentRegistry()
    checkpoint_handler = PostgreSQLAgentCheckpointHandler(SessionLocal)
    agent = LLMAgent(
        agent_definition,
        checkpoint_handler=checkpoint_handler,
    )

    async def register_agent():
        await agent_registry.register(agent)

    import asyncio

    asyncio.run(register_agent())

    return AgentRuntime(
        agent_registry,
        tool_execution_service=execution_service,
        llm_gateway=llm_gateway,
        agent_run_steps_repository_factory=agent_run_steps_repository_factory,
        plan_provider=ApprovalContinuationPlanProvider(),
    )


def test_postgres_approval_continuation_end_to_end_runtime_guarantees() -> None:
    """Prove approval-before-idempotency, durable resume, and exactly-once replay."""

    import asyncio

    session = SessionLocal()

    run_id = str(uuid4())
    step_id = "retrieve_evidence"
    call_id = str(uuid4())
    tool_name = "transfer_funds"
    principal = "api_key:approval-runtime-principal"
    tenant_id = "tenant-approval-runtime"

    request = AgentRequest(
        input="Transfer 500 to acc-123.",
        session_id=f"session-{run_id}",
        user_id="approval-runtime-user",
        principal=principal,
        tenant_id=tenant_id,
        memory_namespace="approval-runtime-memory",
        metadata={
            "source": "approval-runtime-integration",
        },
    )

    arguments = {
        "amount": 500,
        "recipient": "acc-123",
    }

    tool = SideEffectTool(name=tool_name)
    tool_registry = InMemoryToolRegistry()

    authorizer = InMemoryToolAuthorizer()
    authorization_service = ToolAuthorizationService(authorizer)

    idempotency_store = PostgreSQLToolExecutionIdempotencyStore(SessionLocal)
    approval_coordinator = _build_approval_coordinator()

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
        idempotency_store=idempotency_store,
        approval_coordinator=approval_coordinator,
    )

    continuation_gateway = ContinuationLLMGateway()

    agent_definition = __import__(
        "ai_platform.agents.models",
        fromlist=["AgentDefinition"],
    ).AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise approval integration agent.",
        system_prompt="Execute governed enterprise tool calls.",
        model="mock-gpt",
        tool_names=(tool_name,),
    )

    runtime = None

    try:
        asyncio.run(tool_registry.register(tool))

        asyncio.run(
            authorizer.allow(
                principal,
                tool_name,
            )
        )

        run_repository = PostgreSQLAgentRunRepository(session)
        step_repository = PostgreSQLAgentRunStepsRepository(session)
        checkpoint_repository = PostgreSQLAgentCheckpointsRepository(session)
        approval_repository = PostgreSQLApprovalRequestRepository(session)

        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="enterprise-rag-analyst",
                session_id=request.session_id,
                user_id=request.user_id,
                principal=request.principal,
                tenant_id=request.tenant_id,
                status=AgentRunStatus.WAITING_FOR_APPROVAL,
                started_at=datetime.now(UTC),
                request_snapshot=AgentRunRequestSnapshot.from_request(request),
            )
        )

        step_repository.create(
            AgentRunStep(
                run_id=run_id,
                step_id=step_id,
                step_index=0,
                step_type="tool_call",
                status=AgentRunStepStatus.RUNNING,
                attempt=1,
                tool_name=tool_name,
                call_id=call_id,
                input=arguments,
            )
        )

        checkpoint = AgentExecutionCheckpoint(
            schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
            run_id=run_id,
            agent_name="enterprise-rag-analyst",
            session_id=request.session_id,
            user_id=request.user_id,
            messages=(
                user_message(request.input),
                assistant_tool_call_message(
                    tool_calls=(
                        AgentToolCall(
                            call_id=call_id,
                            name=tool_name,
                            arguments=arguments,
                        ),
                    ),
                ),
            ),
            tool_round=1,
            position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
            metadata={
                "source": "approval-runtime-integration",
                "orchestration": {
                    "current_step_index": 0,
                    "steps": {
                        step_id: {
                            "status": "running",
                            "tool_round": 1,
                        },
                    },
                },
            },
        )

        checkpoint_repository.save(checkpoint)

        # ---------------------------------------------------------------
        # 1. INITIAL TOOL ATTEMPT
        #
        # The real ToolExecutionService must stop at approval evaluation.
        # Therefore there must be no idempotency claim yet.
        # ---------------------------------------------------------------

        execution_context = ToolExecutionContext(
            run_id=run_id,
            call_id=call_id,
            agent_name="enterprise-rag-analyst",
            session_id=request.session_id,
            user_id=request.user_id,
            principal=principal,
            tenant_id=tenant_id,
        )

        with pytest.raises(ToolExecutionWaitingForApprovalError):
            asyncio.run(
                execution_service.execute(
                    tool_name,
                    arguments,
                    principal=principal,
                    execution_context=execution_context,
                    step_id=step_id,
                )
            )

        session.rollback()

        approval_rows = session.scalars(
            select(ApprovalRequestRecord).where(
                ApprovalRequestRecord.run_id == run_id,
            )
        ).all()

        assert len(approval_rows) == 1
        assert approval_rows[0].status == ApprovalStatus.PENDING.value
        approval_id = approval_rows[0].approval_id

        idempotency_record = session.scalar(
            select(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == run_id,
                ToolExecutionIdempotencyRecord.call_id == call_id,
                ToolExecutionIdempotencyRecord.tool_name == tool_name,
            )
        )

        assert idempotency_record is None
        assert tool.execution_count == 0

        persisted_checkpoint = checkpoint_repository.get_latest(run_id)

        assert persisted_checkpoint is not None
        assert persisted_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
        assert persisted_checkpoint.messages[-1] == checkpoint.messages[-1]

        orchestration_metadata = persisted_checkpoint.metadata["orchestration"]

        assert orchestration_metadata["current_step_index"] == 0
        assert orchestration_metadata["steps"][step_id]["status"] == "running"

        # ---------------------------------------------------------------
        # 2. BUILD THE REAL RUNTIME
        #
        # The production deterministic plan provider will rebuild:
        #
        #   retrieve_evidence -> analyze_evidence -> produce_answer
        #
        # The restored checkpoint therefore lands on the same
        # retrieve_evidence step.
        # ---------------------------------------------------------------

        step_repositories = []

        def agent_run_steps_repository_factory():
            repository = PostgreSQLAgentRunStepsRepository(SessionLocal())
            step_repositories.append(repository)
            return repository

        runtime = _build_runtime(
            execution_service=execution_service,
            agent_definition=agent_definition,
            llm_gateway=continuation_gateway,
            agent_run_steps_repository_factory=agent_run_steps_repository_factory,
        )

        observer = PostgreSQLAgentRunEventObserver(SessionLocal)

        continuation_service = AgentRunApprovalContinuationService(
            runtime=runtime,
            approval_repository=approval_repository,
            agent_run_repository=run_repository,
            agent_run_steps_repository=step_repository,
            checkpoints_repository=checkpoint_repository,
            lease_seconds=60,
            observer=observer,
        )

        # ---------------------------------------------------------------
        # 3. HUMAN APPROVAL + REAL AGENT RUNTIME RESUME
        # ---------------------------------------------------------------

        response = asyncio.run(
            continuation_service.continue_approval(
                approval_id,
                status=ApprovalStatus.APPROVED,
                resolved_by="approver-runtime-integration",
                resolution_reason="Approved for the integration test.",
            )
        )

        assert response is not None

        # The pending tool call is restored from the durable checkpoint.
        # After the tool completes, the LLM may be queried to synthesize
        # the continuation response.
        assert continuation_gateway.calls == 1
        assert continuation_gateway.requests

        # Exactly one real side effect occurred.
        assert tool.execution_count == 1
        assert tool.executed_arguments == [arguments]

        # ---------------------------------------------------------------
        # 4. DURABLE POSTGRES STATE AFTER RESUME
        # ---------------------------------------------------------------

        session.rollback()

        restored_approval = approval_repository.get(approval_id)

        assert restored_approval is not None
        assert restored_approval.status is ApprovalStatus.APPROVED
        assert restored_approval.resolved_by == "approver-runtime-integration"

        restored_run = run_repository.get(run_id)

        assert restored_run is not None
        assert restored_run.status is AgentRunStatus.COMPLETED
        assert restored_run.lease_id is None
        assert restored_run.lease_expires_at is None

        restored_step = step_repository.get(run_id, step_id)

        assert restored_step is not None
        assert restored_step.status is AgentRunStepStatus.COMPLETED
        assert restored_step.tool_name == tool_name
        assert restored_step.call_id == call_id
        assert restored_step.output == {
            "status": "transferred",
            "amount": 500,
            "recipient": "acc-123",
        }

        completed_idempotency_record = session.scalar(
            select(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == run_id,
                ToolExecutionIdempotencyRecord.call_id == call_id,
                ToolExecutionIdempotencyRecord.tool_name == tool_name,
            )
        )

        assert completed_idempotency_record is not None
        assert completed_idempotency_record.status == "completed"
        assert completed_idempotency_record.success is True

        # ---------------------------------------------------------------
        # 5. REPLAY THE SAME TOOL CALL
        #
        # Same run_id + call_id + tool_name must return the cached result.
        # The side effect must not execute again.
        # ---------------------------------------------------------------

        replay_result = asyncio.run(
            execution_service.execute(
                tool_name,
                arguments,
                principal=principal,
                execution_context=execution_context,
                step_id=step_id,
            )
        )

        assert replay_result.success is True
        assert replay_result.output == {
            "status": "transferred",
            "amount": 500,
            "recipient": "acc-123",
        }

        assert tool.execution_count == 1
        assert tool.executed_arguments == [arguments]

        session.rollback()

        replay_idempotency_record = session.scalar(
            select(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == run_id,
                ToolExecutionIdempotencyRecord.call_id == call_id,
                ToolExecutionIdempotencyRecord.tool_name == tool_name,
            )
        )

        assert replay_idempotency_record is not None
        assert replay_idempotency_record.status == "completed"
        assert replay_idempotency_record.success is True

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
            delete(AgentRunEventRecord).where(
                AgentRunEventRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunStepRecord).where(
                AgentRunStepRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(ApprovalRequestRecord).where(
                ApprovalRequestRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
        session.close()


def test_postgres_approved_continuation_failure_recovers_without_reapproval() -> None:
    """Prove an approved continuation survives failure and recovers exactly once."""

    import asyncio

    session = SessionLocal()

    run_id = str(uuid4())
    step_id = "retrieve_evidence"
    call_id = str(uuid4())
    tool_name = "transfer_funds"
    principal = "api_key:approval-recovery-principal"
    tenant_id = "tenant-approval-recovery"

    request = AgentRequest(
        input="Transfer 500 to acc-123.",
        session_id=f"session-{run_id}",
        user_id="approval-recovery-user",
        principal=principal,
        tenant_id=tenant_id,
        memory_namespace="approval-recovery-memory",
        metadata={
            "source": "approval-recovery-integration",
        },
    )

    arguments = {
        "amount": 500,
        "recipient": "acc-123",
    }

    tool = SideEffectTool(name=tool_name)
    tool_registry = InMemoryToolRegistry()

    authorizer = InMemoryToolAuthorizer()
    authorization_service = ToolAuthorizationService(authorizer)

    idempotency_store = PostgreSQLToolExecutionIdempotencyStore(SessionLocal)
    approval_coordinator = _build_approval_coordinator()

    execution_service = ToolExecutionService(
        tool_registry,
        authorization_service=authorization_service,
        idempotency_store=idempotency_store,
        approval_coordinator=approval_coordinator,
    )

    continuation_gateway = ContinuationLLMGateway()

    agent_definition = __import__(
        "ai_platform.agents.models",
        fromlist=["AgentDefinition"],
    ).AgentDefinition(
        name="enterprise-rag-analyst",
        description="Enterprise approval recovery integration agent.",
        system_prompt="Execute governed enterprise tool calls.",
        model="mock-gpt",
        tool_names=(tool_name,),
    )

    try:
        asyncio.run(tool_registry.register(tool))
        asyncio.run(authorizer.allow(principal, tool_name))

        run_repository = PostgreSQLAgentRunRepository(session)
        step_repository = PostgreSQLAgentRunStepsRepository(session)
        checkpoint_repository = PostgreSQLAgentCheckpointsRepository(session)
        approval_repository = PostgreSQLApprovalRequestRepository(session)

        run_repository.create(
            AgentRun(
                run_id=run_id,
                agent_name="enterprise-rag-analyst",
                session_id=request.session_id,
                user_id=request.user_id,
                principal=request.principal,
                tenant_id=request.tenant_id,
                status=AgentRunStatus.WAITING_FOR_APPROVAL,
                started_at=datetime.now(UTC),
                request_snapshot=AgentRunRequestSnapshot.from_request(request),
            )
        )

        step_repository.create(
            AgentRunStep(
                run_id=run_id,
                step_id=step_id,
                step_index=0,
                step_type="tool_call",
                status=AgentRunStepStatus.RUNNING,
                attempt=1,
                tool_name=tool_name,
                call_id=call_id,
                input=arguments,
            )
        )

        checkpoint = AgentExecutionCheckpoint(
            schema_version=AgentExecutionCheckpoint.CURRENT_SCHEMA_VERSION,
            run_id=run_id,
            agent_name="enterprise-rag-analyst",
            session_id=request.session_id,
            user_id=request.user_id,
            messages=(
                user_message(request.input),
                assistant_tool_call_message(
                    tool_calls=(
                        AgentToolCall(
                            call_id=call_id,
                            name=tool_name,
                            arguments=arguments,
                        ),
                    ),
                ),
            ),
            tool_round=1,
            position=AgentCheckpointPosition.BEFORE_TOOL_EXECUTION,
            metadata={
                "source": "approval-recovery-integration",
                "orchestration": {
                    "current_step_index": 0,
                    "steps": {
                        step_id: {
                            "status": "running",
                            "tool_round": 1,
                        },
                    },
                },
            },
        )

        checkpoint_repository.save(checkpoint)

        # ---------------------------------------------------------------
        # 1. CREATE THE REAL PENDING APPROVAL
        # ---------------------------------------------------------------

        execution_context = ToolExecutionContext(
            run_id=run_id,
            call_id=call_id,
            agent_name="enterprise-rag-analyst",
            session_id=request.session_id,
            user_id=request.user_id,
            principal=principal,
            tenant_id=tenant_id,
        )

        with pytest.raises(ToolExecutionWaitingForApprovalError):
            asyncio.run(
                execution_service.execute(
                    tool_name,
                    arguments,
                    principal=principal,
                    execution_context=execution_context,
                    step_id=step_id,
                )
            )

        session.rollback()

        approval_rows = session.scalars(
            select(ApprovalRequestRecord).where(
                ApprovalRequestRecord.run_id == run_id,
            )
        ).all()

        assert len(approval_rows) == 1
        assert approval_rows[0].status == ApprovalStatus.PENDING.value
        approval_id = approval_rows[0].approval_id

        assert tool.execution_count == 0

        idempotency_record = session.scalar(
            select(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == run_id,
                ToolExecutionIdempotencyRecord.call_id == call_id,
                ToolExecutionIdempotencyRecord.tool_name == tool_name,
            )
        )
        assert idempotency_record is None

        # ---------------------------------------------------------------
        # 2. APPROVE, THEN CRASH DURING CONTINUATION
        #
        # The approval is committed before runtime.resume() is invoked.
        # The crashing runtime therefore exercises the precise durable
        # failure boundary we want to protect.
        # ---------------------------------------------------------------

        crash_runtime = CrashAfterApprovalRuntime()

        continuation_service = AgentRunApprovalContinuationService(
            runtime=crash_runtime,
            approval_repository=approval_repository,
            agent_run_repository=run_repository,
            agent_run_steps_repository=step_repository,
            checkpoints_repository=checkpoint_repository,
            lease_seconds=60,
        )

        with pytest.raises(
            RuntimeError,
            match="simulated approval continuation crash",
        ):
            asyncio.run(
                continuation_service.continue_approval(
                    approval_id,
                    status=ApprovalStatus.APPROVED,
                    resolved_by="approver-recovery-integration",
                    resolution_reason="Approved for recovery integration test.",
                )
            )

        session.rollback()

        # Approval must remain durably APPROVED even though continuation
        # failed after acquiring the run lease.
        failed_approval = approval_repository.get(approval_id)

        assert failed_approval is not None
        assert failed_approval.status is ApprovalStatus.APPROVED
        assert failed_approval.resolved_by == "approver-recovery-integration"

        failed_run = run_repository.get(run_id)

        assert failed_run is not None
        assert failed_run.status is AgentRunStatus.FAILED
        assert failed_run.lease_id is None
        assert failed_run.lease_expires_at is None
        assert failed_run.error_type == "RuntimeError"
        assert "simulated approval continuation crash" in (failed_run.error_message or "")

        # The crash happened before ToolExecutionService was reached.
        assert tool.execution_count == 0

        idempotency_record = session.scalar(
            select(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == run_id,
                ToolExecutionIdempotencyRecord.call_id == call_id,
                ToolExecutionIdempotencyRecord.tool_name == tool_name,
            )
        )
        assert idempotency_record is None

        durable_checkpoint = checkpoint_repository.get_latest(run_id)

        assert durable_checkpoint is not None
        assert durable_checkpoint.position is AgentCheckpointPosition.BEFORE_TOOL_EXECUTION
        assert durable_checkpoint.messages[-1] == checkpoint.messages[-1]

        # ---------------------------------------------------------------
        # 3. RECOVER THE FAILED RUN WITH THE REAL AGENT RUNTIME
        #
        # Recovery must use the existing APPROVED approval. It must not
        # create another approval request before executing the tool.
        # ---------------------------------------------------------------

        step_repositories = []

        def agent_run_steps_repository_factory():
            repository = PostgreSQLAgentRunStepsRepository(SessionLocal())
            step_repositories.append(repository)
            return repository

        recovery_runtime = _build_runtime(
            execution_service=execution_service,
            agent_definition=agent_definition,
            llm_gateway=continuation_gateway,
            agent_run_steps_repository_factory=agent_run_steps_repository_factory,
        )

        observer = PostgreSQLAgentRunEventObserver(SessionLocal)

        recovery_service = AgentRunRecoveryService(
            runtime=recovery_runtime,
            repository=run_repository,
            checkpoints_repository=checkpoint_repository,
            observer=observer,
            tool_idempotency_store=idempotency_store,
            lease_seconds=60,
        )

        recovery_result = asyncio.run(recovery_service.recover(run_id))

        assert recovery_result.run_id == run_id
        assert recovery_result.response.output == ("Transfer completed successfully.")

        # The existing approval must have been reused.
        restored_approval = approval_repository.get(approval_id)

        assert restored_approval is not None
        assert restored_approval.status is ApprovalStatus.APPROVED
        assert restored_approval.resolved_by == "approver-recovery-integration"

        restored_approval_rows = session.scalars(
            select(ApprovalRequestRecord).where(
                ApprovalRequestRecord.run_id == run_id,
            )
        ).all()

        assert len(restored_approval_rows) == 1

        # Exactly one real side effect occurred during recovery.
        assert tool.execution_count == 1
        assert tool.executed_arguments == [arguments]

        # ---------------------------------------------------------------
        # 4. VERIFY DURABLE RECOVERY STATE
        # ---------------------------------------------------------------

        session.rollback()

        restored_run = run_repository.get(run_id)

        assert restored_run is not None
        assert restored_run.status is AgentRunStatus.COMPLETED
        assert restored_run.lease_id is None
        assert restored_run.lease_expires_at is None
        assert restored_run.output == "Transfer completed successfully."

        restored_step = step_repository.get(run_id, step_id)

        assert restored_step is not None
        assert restored_step.status is AgentRunStepStatus.COMPLETED
        assert restored_step.tool_name == tool_name
        assert restored_step.call_id == call_id
        assert restored_step.output == {
            "status": "transferred",
            "amount": 500,
            "recipient": "acc-123",
        }

        completed_idempotency_record = session.scalar(
            select(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == run_id,
                ToolExecutionIdempotencyRecord.call_id == call_id,
                ToolExecutionIdempotencyRecord.tool_name == tool_name,
            )
        )

        assert completed_idempotency_record is not None
        assert completed_idempotency_record.status == "completed"
        assert completed_idempotency_record.success is True

        restored_checkpoint = checkpoint_repository.get_latest(run_id)

        assert restored_checkpoint is not None
        assert restored_checkpoint.position is AgentCheckpointPosition.AFTER_TOOL_EXECUTION

        assert any(
            message.role.value == "tool" and call_id in message.content
            for message in restored_checkpoint.messages
        )

        # Only the final-answer LLM call is expected. The pending tool call
        # itself came from the durable BEFORE_TOOL_EXECUTION checkpoint.
        assert continuation_gateway.calls == 1
        assert continuation_gateway.requests

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
            delete(AgentRunEventRecord).where(
                AgentRunEventRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunStepRecord).where(
                AgentRunStepRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(ApprovalRequestRecord).where(
                ApprovalRequestRecord.run_id == run_id,
            )
        )
        session.execute(
            delete(AgentRunRecord).where(
                AgentRunRecord.run_id == run_id,
            )
        )
        session.commit()
        session.close()
