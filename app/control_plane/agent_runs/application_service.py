from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from ai_platform.agents.budget import ExecutionBudget
from ai_platform.agents.exceptions import (
    AgentExecutionControlSignal,
    AgentExecutionOwnershipLostError,
)
from ai_platform.agents.models import AgentRequest, AgentResponse
from ai_platform.agents.policy import TenantPolicy, TenantPolicyEngine
from rag.governance import GovernancePolicy
from ai_platform.agents.observability import (
    AgentExecutionEvent,
    AgentExecutionEventType,
)
from ai_platform.agents.observer import AgentExecutionObserver
from ai_platform.agents.runtime import AgentRuntime
from app.control_plane.agent_runs.cancellation import (
    AgentRunCancellationRegistry,
)
from app.control_plane.agent_runs.admission import (
    AgentRunAdmissionPolicy,
    AllowAllAgentRunAdmissionPolicy,
)
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
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot
from app.control_plane.agent_run_events.repository import (
    AgentRunEventsRepository,
)
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_run_steps.repository import (
    AgentRunStepsRepository,
)
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.lease import (
    create_lease,
    heartbeat_loop,
)


class AgentRunApplicationService:
    def __init__(
        self,
        *,
        runtime: AgentRuntime,
        repository: AgentRunRepository,
        events_repository: AgentRunEventsRepository | None = None,
        agent_run_steps_repository: AgentRunStepsRepository | None = None,
        observer: AgentExecutionObserver | None = None,
        admission_policy: AgentRunAdmissionPolicy | None = None,
        cancellation_registry: AgentRunCancellationRegistry | None = None,
        tenant_policy_engine: TenantPolicyEngine | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self._runtime = runtime
        self._repository = repository
        self._events_repository = events_repository
        self._agent_run_steps_repository = agent_run_steps_repository
        self._observer = observer
        self._admission_policy = (
            admission_policy if admission_policy is not None else AllowAllAgentRunAdmissionPolicy()
        )
        self._lease_seconds = lease_seconds
        self._cancellation_registry = cancellation_registry
        self._tenant_policy_engine = tenant_policy_engine

    async def _emit_governance_decision(
        self,
        *,
        run: AgentRun,
        governance_domain: str,
        decision: str,
        reason: str | None = None,
        policy_id: str | None = None,
        policy_version: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        if self._observer is None:
            return

        metadata: dict[str, object] = {
            "governance_domain": governance_domain,
            "decision": decision,
        }

        if run.tenant_id is not None:
            metadata["tenant_id"] = run.tenant_id

        if reason is not None:
            metadata["reason"] = reason

        if policy_id is not None:
            metadata["policy_id"] = policy_id

        if policy_version is not None:
            metadata["policy_version"] = policy_version

        if details:
            metadata["details"] = dict(details)

        await self._emit(
            AgentExecutionEvent(
                event_type=AgentExecutionEventType.GOVERNANCE_DECISION,
                agent_name=run.agent_name,
                run_id=run.run_id,
                session_id=run.session_id,
                user_id=run.user_id,
                metadata=metadata,
            )
        )

    async def _emit_model_governance_decision(
        self,
        *,
        run: AgentRun,
        decision,
    ) -> None:
        if decision is None:
            return

        await self._emit_governance_decision(
            run=run,
            governance_domain="model",
            decision="allow",
            policy_id=decision.policy_id,
            policy_version=decision.policy_version,
            details={
                "effective_model": decision.effective_model,
                "effective_provider": decision.effective_provider,
            },
        )

    async def _resolve_model_governance(
        self,
        *,
        agent_name: str,
        tenant_id: str | None,
        requested_decision,
    ):
        if requested_decision is not None:
            return requested_decision

        if tenant_id is None or self._tenant_policy_engine is None:
            return None

        agent_definition = await self._runtime.get_agent_definition(agent_name)
        model = agent_definition.llm_config.model

        if model is None:
            raise ValueError(
                f"Agent '{agent_name}' does not define an LLM model "
                "required for model governance."
            )

        return self._tenant_policy_engine.authorize_model(
            tenant_id=tenant_id,
            model=model,
        )

    async def _authorize_memory_namespace(
        self,
        *,
        agent_name: str,
        tenant_id: str | None,
        memory_namespace: str | None,
    ) -> None:
        if tenant_id is None or memory_namespace is None or self._tenant_policy_engine is None:
            return

        self._tenant_policy_engine.authorize_memory_namespace(
            tenant_id=tenant_id,
            memory_namespace=memory_namespace,
            operation="read",
        )

        agent_definition = await self._runtime.get_agent_definition(agent_name)

        if agent_definition.memory_write_enabled:
            self._tenant_policy_engine.authorize_memory_namespace(
                tenant_id=tenant_id,
                memory_namespace=memory_namespace,
                operation="write",
            )

    def _resolve_effective_governance_policy(
        self,
        *,
        tenant_id: str | None,
        requested_policy: GovernancePolicy | None,
    ) -> GovernancePolicy | None:
        if tenant_id is None:
            return requested_policy

        policy = (
            self._tenant_policy_engine.get_policy(tenant_id)
            if self._tenant_policy_engine is not None
            else None
        )

        if policy is not None and policy.allow_cross_tenant_data:
            return requested_policy

        base_policy = requested_policy if requested_policy is not None else GovernancePolicy()

        return base_policy.with_tenant_scope(tenant_id)

    def _resolve_effective_budget(
        self,
        *,
        tenant_id: str | None,
        requested_budget: ExecutionBudget,
    ) -> tuple[ExecutionBudget, TenantPolicy | None]:
        if self._tenant_policy_engine is None or not tenant_id:
            return requested_budget, None

        policy = self._tenant_policy_engine.get_policy(tenant_id)

        effective_max_tokens = requested_budget.max_tokens_per_run
        if policy.max_tokens_per_run is not None:
            if effective_max_tokens is None:
                effective_max_tokens = policy.max_tokens_per_run
            else:
                effective_max_tokens = min(
                    effective_max_tokens,
                    policy.max_tokens_per_run,
                )

        effective_budget = ExecutionBudget(
            max_llm_calls=requested_budget.max_llm_calls,
            max_tool_calls=requested_budget.max_tool_calls,
            max_tool_rounds=requested_budget.max_tool_rounds,
            max_duration_seconds=requested_budget.max_duration_seconds,
            max_tokens_per_run=effective_max_tokens,
        )

        return effective_budget, policy

    async def _emit(
        self,
        event: AgentExecutionEvent,
    ) -> None:
        if self._observer is None:
            return

        try:
            await self._observer.record(event)
        except Exception:
            # Cancellation observability must never change cancellation semantics.
            pass

    @staticmethod
    def _build_replayed_response(
        run: AgentRun,
    ) -> AgentResponse:
        return AgentResponse(
            agent_name=run.agent_name,
            output=run.output,
            session_id=run.session_id,
            metadata=dict(run.metadata),
        )

    @staticmethod
    def _request_snapshots_match(
        existing_run: AgentRun,
        request: AgentRequest,
    ) -> bool:
        if existing_run.request_snapshot is None:
            return False

        if existing_run.session_id != request.session_id:
            return False

        return existing_run.request_snapshot == AgentRunRequestSnapshot.from_request(request)

    def _resolve_existing_idempotent_run(
        self,
        *,
        existing_run: AgentRun,
        agent_name: str,
        request: AgentRequest,
    ) -> AgentRunExecutionResult:
        if existing_run.agent_name != agent_name:
            raise AgentRunIdempotencyConflictError(
                "Idempotency key is already associated with a different agent run."
            )

        if not self._request_snapshots_match(existing_run, request):
            raise AgentRunIdempotencyConflictError(
                "Idempotency key is already associated with a different request."
            )

        if existing_run.status is AgentRunStatus.FAILED:
            message = existing_run.error_message or "Agent run failed."
            raise RuntimeError(
                f"Idempotent agent run '{existing_run.run_id}' previously failed: {message}"
            )

        return AgentRunExecutionResult(
            run_id=existing_run.run_id,
            response=self._build_replayed_response(existing_run),
        )

    async def execute(
        self,
        *,
        agent_name: str,
        request: AgentRequest,
        idempotency_key: str | None = None,
    ) -> AgentRunExecutionResult:
        effective_budget, budget_policy = self._resolve_effective_budget(
            tenant_id=request.tenant_id,
            requested_budget=request.execution_budget or ExecutionBudget(),
        )
        effective_governance_policy = self._resolve_effective_governance_policy(
            tenant_id=request.tenant_id,
            requested_policy=request.governance_policy,
        )
        effective_model_governance = await self._resolve_model_governance(
            agent_name=agent_name,
            tenant_id=request.tenant_id,
            requested_decision=request.model_governance,
        )
        effective_request = replace(
            request,
            execution_budget=effective_budget,
            governance_policy=effective_governance_policy,
            model_governance=effective_model_governance,
        )

        await self._authorize_memory_namespace(
            agent_name=agent_name,
            tenant_id=effective_request.tenant_id,
            memory_namespace=effective_request.memory_namespace,
        )

        if idempotency_key is not None:
            idempotency_key = idempotency_key.strip()

            if not idempotency_key:
                raise ValueError("Idempotency key must not be empty when provided.")

            if effective_request.user_id is None:
                raise ValueError("A user_id is required when an idempotency key is provided.")

            existing_run = self._repository.get_by_idempotency_key(
                effective_request.tenant_id,
                effective_request.user_id,
                idempotency_key,
            )

            if existing_run is not None:
                return self._resolve_existing_idempotent_run(
                    existing_run=existing_run,
                    agent_name=agent_name,
                    request=effective_request,
                )

        run_id = str(uuid4())

        run = AgentRun(
            run_id=run_id,
            agent_name=agent_name,
            root_run_id=run_id,
            session_id=effective_request.session_id,
            user_id=effective_request.user_id,
            principal=effective_request.principal,
            tenant_id=effective_request.tenant_id,
            idempotency_key=idempotency_key,
            status=AgentRunStatus.PENDING,
            metadata=dict(effective_request.metadata),
            request_snapshot=AgentRunRequestSnapshot.from_request(effective_request),
        )

        try:
            self._repository.create(run)
        except DuplicateAgentRunError:
            if idempotency_key is None or effective_request.user_id is None:
                raise

            existing_run = self._repository.get_by_idempotency_key(
                effective_request.tenant_id,
                effective_request.user_id,
                idempotency_key,
            )

            if existing_run is None:
                raise RuntimeError(
                    "Agent run creation conflicted, but the idempotent run " "could not be loaded."
                )

            return self._resolve_existing_idempotent_run(
                existing_run=existing_run,
                agent_name=agent_name,
                request=effective_request,
            )

        await self._emit_model_governance_decision(
            run=run,
            decision=effective_model_governance,
        )

        if budget_policy is not None:
            await self._emit_governance_decision(
                run=run,
                governance_domain="budget",
                decision="allow",
                policy_id=budget_policy.policy_id,
                policy_version=budget_policy.policy_version,
                details={
                    "max_tokens_per_run": effective_budget.max_tokens_per_run,
                },
            )

        admission = await self._admission_policy.evaluate(
            agent_name=agent_name,
            request=effective_request,
            run=run,
        )

        admission_reason = admission.reason
        await self._emit_governance_decision(
            run=run,
            governance_domain="admission",
            decision="allow" if admission.allowed else "deny",
            reason=admission_reason,
        )

        if not admission.allowed:
            rejected_at = datetime.now(UTC)
            rejected_run = run.transition_to(AgentRunStatus.REJECTED).model_copy(
                update={
                    "completed_at": rejected_at,
                    "metadata": {
                        **run.metadata,
                        "admission": {
                            "allowed": False,
                            "reason": admission.reason,
                        },
                    },
                }
            )

            self._repository.update(rejected_run)

            reason = admission.reason or "Agent run admission was rejected."
            raise AgentRunAdmissionRejectedError(reason)

        started_at = datetime.now(UTC)
        lease_id, lease_expires_at = create_lease(self._lease_seconds)

        run = run.transition_to(AgentRunStatus.RUNNING).model_copy(
            update={
                "started_at": started_at,
                "lease_id": lease_id,
                "lease_expires_at": lease_expires_at,
            }
        )

        self._repository.update(run)

        ownership_lost = asyncio.Event()
        cancellation_requested = asyncio.Event()

        heartbeat_task = asyncio.create_task(
            heartbeat_loop(
                self._repository,
                run_id=run.run_id,
                lease_id=lease_id,
                lease_seconds=self._lease_seconds,
                ownership_lost=ownership_lost,
            )
        )

        execution_task = asyncio.current_task()
        if execution_task is None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            raise RuntimeError("Agent run execution is not attached to an asyncio task.")

        if self._cancellation_registry is not None:
            self._cancellation_registry.register(
                run.run_id,
                execution_task,
                cancellation_requested,
            )

        try:
            current_run = self._repository.get(run.run_id)
            if current_run is not None and current_run.cancellation_requested:
                cancellation_requested.set()

            if cancellation_requested.is_set():
                raise asyncio.CancelledError("Agent run cancellation requested before execution.")

            response = await self._runtime.run(
                agent_name,
                effective_request,
                lease_id=lease_id,
                run_id=run.run_id,
                execution_ownership_lost=ownership_lost,
                cancellation_requested=cancellation_requested,
            )
        except asyncio.CancelledError:
            cancelled_at = datetime.now(UTC)

            try:
                cancelled_run = self._repository.cancel_if_owner(
                    run.run_id,
                    lease_id=lease_id,
                    completed_at=cancelled_at,
                )
            except Exception:
                # Cancellation must continue to propagate even if terminal
                # cancellation persistence fails.
                pass
            else:
                if cancelled_run is not None:
                    await self._emit(
                        AgentExecutionEvent(
                            event_type=AgentExecutionEventType.AGENT_CANCELLED,
                            agent_name=run.agent_name,
                            run_id=run.run_id,
                            session_id=run.session_id,
                            user_id=run.user_id,
                        )
                    )

            raise
        except AgentExecutionOwnershipLostError:
            raise
        except AgentExecutionControlSignal:
            paused_at = datetime.now(UTC)

            paused_run = self._repository.transition_to_waiting_for_approval_if_owner(
                run.run_id,
                lease_id=lease_id,
                updated_at=paused_at,
            )

            if paused_run is None:
                raise RuntimeError(
                    f"Agent run '{run.run_id}' lost lease ownership while pausing for approval."
                )

            raise
        except Exception as exc:
            failed_at = datetime.now(UTC)

            try:
                persisted_failed_run = self._repository.fail_if_owner(
                    run.run_id,
                    lease_id=lease_id,
                    completed_at=failed_at,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            except Exception:
                # Preserve the original execution failure if terminal
                # failure persistence itself fails.
                pass
            else:
                if persisted_failed_run is None:
                    raise RuntimeError(
                        f"Agent run '{run.run_id}' lost lease ownership while failing."
                    ) from exc

            raise
        finally:
            if self._cancellation_registry is not None:
                self._cancellation_registry.unregister(
                    run.run_id,
                    execution_task,
                )

            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

        completed_at = datetime.now(UTC)
        completed_run = self._repository.complete_if_owner(
            run.run_id,
            lease_id=lease_id,
            completed_at=completed_at,
            output=response.output,
        )

        if completed_run is None:
            raise RuntimeError(f"Agent run '{run.run_id}' lost lease ownership before completion.")

        return AgentRunExecutionResult(
            run_id=completed_run.run_id,
            response=response,
        )

    def get_run(
        self,
        run_id: str,
        *,
        tenant_id: str | None = None,
        principal: str | None = None,
    ) -> AgentRun | None:
        self._validate_identity_context(
            tenant_id=tenant_id,
            principal=principal,
        )

        if tenant_id is None and principal is None:
            return self._repository.get(run_id)

        run = self._repository.get_for_tenant(run_id, tenant_id)

        if run is None:
            return None

        self._authorize_run_access(
            run,
            tenant_id=tenant_id,
            principal=principal,
        )

        return run

    def cancel(
        self,
        run_id: str,
        *,
        tenant_id: str | None = None,
        principal: str | None = None,
    ) -> AgentRun:
        self._validate_identity_context(
            tenant_id=tenant_id,
            principal=principal,
        )

        if tenant_id is None and principal is None:
            run = self._repository.get(run_id)
        else:
            run = self._repository.get_for_tenant(run_id, tenant_id)

            if run is not None:
                self._authorize_run_access(
                    run,
                    tenant_id=tenant_id,
                    principal=principal,
                )

        if run is None:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        if run.status is not AgentRunStatus.RUNNING:
            raise ValueError(
                f"Agent run '{run_id}' is not cancellable from status " f"'{run.status.value}'.",
            )

        requested = self._repository.request_cancellation(
            run_id,
            requested_at=datetime.now(UTC),
        )

        if requested is None:
            raise RuntimeError(
                f"Agent run '{run_id}' could not accept cancellation.",
            )

        if self._cancellation_registry is not None:
            self._cancellation_registry.cancel(run_id)

        return requested

    def list_runs(
        self,
        *,
        tenant_id: str | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]:
        return self._repository.list(
            tenant_id=tenant_id,
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            status=status,
            limit=limit,
        )

    def list_events(
        self,
        run_id: str,
        *,
        tenant_id: str | None = None,
        principal: str | None = None,
        limit: int = 100,
    ) -> list[AgentExecutionEvent]:
        self._validate_identity_context(
            tenant_id=tenant_id,
            principal=principal,
        )

        if tenant_id is None and principal is None:
            run = self._repository.get(run_id)
        else:
            run = self._repository.get_for_tenant(run_id, tenant_id)

            if run is not None:
                self._authorize_run_access(
                    run,
                    tenant_id=tenant_id,
                    principal=principal,
                )

        if run is None:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        if self._events_repository is None:
            raise RuntimeError(
                "Agent run events repository is not configured.",
            )

        return self._events_repository.list(
            run_id,
            limit=limit,
        )

    @staticmethod
    def _validate_identity_context(
        *,
        tenant_id: str | None,
        principal: str | None,
    ) -> None:
        if (tenant_id is None) != (principal is None):
            raise AgentRunAccessDeniedError(
                "Both tenant_id and principal are required for scoped access.",
            )

    def _authorize_run_access(
        self,
        run: AgentRun,
        *,
        tenant_id: str | None,
        principal: str | None,
    ) -> None:
        if (
            tenant_id is None
            or run.tenant_id != tenant_id
            or principal is None
            or run.principal != principal
        ):
            raise AgentRunAccessDeniedError(
                f"Principal is not authorized to access agent run '{run.run_id}'.",
            )

    def list_steps(
        self,
        run_id: str,
        *,
        tenant_id: str | None,
        principal: str | None,
        status: AgentRunStepStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRunStep]:
        run = self._repository.get_for_tenant(run_id, tenant_id)

        if run is None:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        self._authorize_run_access(
            run,
            tenant_id=tenant_id,
            principal=principal,
        )

        if self._agent_run_steps_repository is None:
            raise RuntimeError(
                "Agent run steps repository is not configured.",
            )

        return self._agent_run_steps_repository.list(
            run_id,
            status=status,
            limit=limit,
        )

    def get_step(
        self,
        run_id: str,
        step_id: str,
        *,
        tenant_id: str,
        principal: str | None,
    ) -> AgentRunStep | None:
        run = self._repository.get_for_tenant(run_id, tenant_id)

        if run is None:
            raise LookupError(
                f"Agent run '{run_id}' was not found.",
            )

        self._authorize_run_access(
            run,
            tenant_id=tenant_id,
            principal=principal,
        )

        if self._agent_run_steps_repository is None:
            raise RuntimeError(
                "Agent run steps repository is not configured.",
            )

        return self._agent_run_steps_repository.get(
            run_id,
            step_id,
        )
