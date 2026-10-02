from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid4, uuid5

from ai_platform.agents.contracts import AgentRegistry
from ai_platform.agents.exceptions import AgentExecutionWaitingForApprovalError
from ai_platform.agents.models import AgentRequest
from app.control_plane.agent_delegation.models import (
    AgentDelegationRequest,
    AgentDelegationResult,
)
from app.control_plane.agent_delegation.policy import AgentDelegationPolicy
from app.control_plane.agent_run_steps.models import (
    AgentRunStep,
    AgentRunStepStatus,
)
from app.control_plane.agent_runs.exceptions import (
    AgentRunAlreadyExecutingError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot


class AgentDelegationService:
    """
    Control-plane primitive for creating durable child-agent delegations.

    Child execution is optional and, when configured, executes the
    already-persisted child run so its durable identity and hierarchy remain
    intact.
    """

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_run_repository: AgentRunRepository,
        agent_run_steps_repository_factory,
        agent_run_application_service=None,
        policy: AgentDelegationPolicy | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._agent_run_repository = agent_run_repository
        self._agent_run_steps_repository_factory = agent_run_steps_repository_factory
        self._agent_run_application_service = agent_run_application_service
        self._policy = policy or AgentDelegationPolicy()

    def get_parent_run(self, run_id: str):
        """Return the durable run used as the delegation parent."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string.")

        return self._agent_run_repository.get(run_id)

    async def delegate(
        self,
        request: AgentDelegationRequest,
    ) -> AgentDelegationResult:
        parent = self._agent_run_repository.get(request.parent_run_id)

        if parent is None:
            raise ValueError("parent agent run does not exist: " f"{request.parent_run_id}")

        if parent.status not in {
            AgentRunStatus.RUNNING,
            AgentRunStatus.WAITING_FOR_APPROVAL,
        }:
            raise ValueError(
                "agent delegation requires an active parent run: " f"{parent.status.value}"
            )

        self._policy.validate_target(request.child_agent_name)

        target = await self._agent_registry.get(request.child_agent_name)

        if target is None:
            raise ValueError(
                "agent delegation target does not exist: " f"{request.child_agent_name}"
            )

        if not target.definition.enabled:
            raise ValueError("agent delegation target is disabled: " f"{request.child_agent_name}")

        depth = self._delegation_depth(parent)
        child_depth = depth + 1
        self._policy.validate_depth(child_depth)

        child_request = self._inherit_identity(
            parent,
            request.child_request,
        )

        idempotency_key = self._normalize_idempotency_key(request.idempotency_key)

        durable_idempotency_key = self._build_idempotency_key(
            parent_run_id=parent.run_id,
            idempotency_key=idempotency_key,
        )

        if durable_idempotency_key is not None:
            if child_request.user_id is None:
                raise ValueError(
                    "child request user_id is required when an idempotency " "key is provided."
                )

            existing = self._agent_run_repository.get_by_idempotency_key(
                child_request.tenant_id,
                child_request.user_id,
                durable_idempotency_key,
            )

            if existing is not None:
                self._validate_existing_child(
                    existing,
                    parent=parent,
                    child_agent_name=request.child_agent_name,
                )

                return AgentDelegationResult(
                    child_run=existing,
                    parent_step_id=existing.parent_step_id or "",
                    created=False,
                )

        step_id = self._build_step_id(
            parent_run_id=parent.run_id,
            idempotency_key=idempotency_key,
        )

        steps_repository = self._agent_run_steps_repository_factory()

        try:
            existing_step = steps_repository.get(
                parent.run_id,
                step_id,
            )

            if existing_step is None:
                steps = steps_repository.list(parent.run_id)

                next_index = (
                    max(
                        (step.step_index for step in steps),
                        default=-1,
                    )
                    + 1
                )

                steps_repository.create(
                    AgentRunStep(
                        run_id=parent.run_id,
                        step_id=step_id,
                        step_index=next_index,
                        step_type="delegation",
                        status=AgentRunStepStatus.PLANNED,
                        input={
                            "agent_name": request.child_agent_name,
                            "request": child_request.input,
                        },
                        metadata={
                            "child_agent_name": request.child_agent_name,
                            "delegation_depth": child_depth,
                            **request.metadata,
                        },
                    )
                )
            elif existing_step.step_type != "delegation":
                raise ValueError(
                    "delegation step id collides with a non-delegation step: " f"{step_id}"
                )
        finally:
            steps_repository.close()

        child_run_id = self._build_child_run_id(
            parent_run_id=parent.run_id,
            parent_step_id=step_id,
            idempotency_key=idempotency_key,
        )

        causation_id = request.causation_id or f"{parent.run_id}:{step_id}"

        child_metadata = dict(child_request.metadata)
        child_metadata.update(request.metadata)

        parent_retrieval_artifact = parent.metadata.get("rag_retriever_artifact")
        if parent_retrieval_artifact is not None:
            child_metadata["rag_retriever_artifact"] = parent_retrieval_artifact

        child_metadata["delegation"] = {
            "parent_run_id": parent.run_id,
            "parent_step_id": step_id,
            "root_run_id": parent.root_run_id,
            "delegation_depth": child_depth,
        }

        child_request = replace(
            child_request,
            metadata=child_metadata,
        )

        child_run = AgentRun(
            run_id=child_run_id,
            agent_name=request.child_agent_name,
            root_run_id=parent.root_run_id,
            parent_run_id=parent.run_id,
            parent_step_id=step_id,
            causation_id=causation_id,
            session_id=child_request.session_id,
            user_id=child_request.user_id,
            principal=child_request.principal,
            tenant_id=child_request.tenant_id,
            idempotency_key=durable_idempotency_key,
            status=AgentRunStatus.PENDING,
            metadata=child_metadata,
            request_snapshot=AgentRunRequestSnapshot.from_request(child_request),
        )

        try:
            self._agent_run_repository.create(child_run)
        except DuplicateAgentRunError:
            if durable_idempotency_key is None or child_request.user_id is None:
                raise

            existing = self._agent_run_repository.get_by_idempotency_key(
                child_request.tenant_id,
                child_request.user_id,
                durable_idempotency_key,
            )

            if existing is None:
                raise

            self._validate_existing_child(
                existing,
                parent=parent,
                child_agent_name=request.child_agent_name,
            )

            return AgentDelegationResult(
                child_run=existing,
                parent_step_id=existing.parent_step_id or "",
                created=False,
            )

        return AgentDelegationResult(
            child_run=child_run,
            parent_step_id=step_id,
            created=True,
        )

    async def execute_delegation(
        self,
        request: AgentDelegationRequest,
    ) -> AgentDelegationResult:
        if self._agent_run_application_service is None:
            raise RuntimeError(
                "agent delegation execution requires an " "AgentRunApplicationService."
            )

        result = await self.delegate(request)

        child = self._agent_run_repository.get(result.child_run.run_id)
        if child is None:
            raise RuntimeError(f"delegated child run disappeared: {result.child_run.run_id}")

        steps_repository = self._agent_run_steps_repository_factory()
        try:
            step = steps_repository.get(
                request.parent_run_id,
                result.parent_step_id,
            )

            if step is None:
                raise RuntimeError(
                    "delegation parent step does not exist: "
                    f"{request.parent_run_id}/{result.parent_step_id}"
                )

            if step.step_type != "delegation":
                raise RuntimeError(
                    "delegation parent step has unexpected type: " f"{step.step_type}"
                )

            if step.status in {
                AgentRunStepStatus.COMPLETED,
                AgentRunStepStatus.FAILED,
            }:
                if child.status in {
                    AgentRunStatus.COMPLETED,
                    AgentRunStatus.FAILED,
                }:
                    return result

                raise RuntimeError(
                    "delegation step is terminal while its child run " "is not terminal."
                )

            if step.status is AgentRunStepStatus.PLANNED:
                now = datetime.now(UTC)
                step = steps_repository.transition(
                    request.parent_run_id,
                    result.parent_step_id,
                    status=AgentRunStepStatus.RUNNING,
                    updated_at=now,
                    started_at=now,
                )

                if step is None:
                    raise RuntimeError("delegation parent step disappeared before execution.")

            if step.status is not AgentRunStepStatus.RUNNING:
                raise RuntimeError(
                    "delegation execution requires a PLANNED or RUNNING "
                    "parent step: "
                    f"{step.status.value}"
                )
        finally:
            steps_repository.close()

        if child.status in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
        }:
            self._reconcile_completed_child(
                request.parent_run_id,
                result.parent_step_id,
                child,
            )
            return AgentDelegationResult(
                child_run=child,
                parent_step_id=result.parent_step_id,
                created=result.created,
            )

        if child.status in {
            AgentRunStatus.RUNNING,
            AgentRunStatus.WAITING_FOR_APPROVAL,
        }:
            return AgentDelegationResult(
                child_run=child,
                parent_step_id=result.parent_step_id,
                created=result.created,
            )

        if child.status is not AgentRunStatus.PENDING:
            raise RuntimeError(
                "delegated child execution requires PENDING status: " f"{child.status.value}"
            )

        if child.request_snapshot is None:
            raise RuntimeError(f"delegated child run '{child.run_id}' has no request snapshot.")

        child_request = child.request_snapshot.to_request(
            session_id=child.session_id,
            user_id=child.user_id,
            principal=child.principal,
        )

        try:
            await self._agent_run_application_service.execute_existing_run(
                run_id=child.run_id,
                agent_name=child.agent_name,
                request=child_request,
            )
        except AgentExecutionWaitingForApprovalError:
            waiting_child = self._agent_run_repository.get(child.run_id)

            if waiting_child is None:
                raise RuntimeError(f"delegated child run disappeared: {child.run_id}")

            return AgentDelegationResult(
                child_run=waiting_child,
                parent_step_id=result.parent_step_id,
                created=result.created,
            )
        except AgentRunAlreadyExecutingError:
            running_child = self._agent_run_repository.get(child.run_id)

            if running_child is None:
                raise RuntimeError(f"delegated child run disappeared: {child.run_id}")

            if running_child.status is not AgentRunStatus.RUNNING:
                raise RuntimeError(
                    "delegated child reported concurrent execution but is no longer "
                    f"RUNNING: {running_child.status.value}"
                )

            return AgentDelegationResult(
                child_run=running_child,
                parent_step_id=result.parent_step_id,
                created=result.created,
            )
        except Exception as exc:
            failed_child = self._agent_run_repository.get(child.run_id)

            steps_repository = self._agent_run_steps_repository_factory()
            try:
                current_step = steps_repository.get(
                    request.parent_run_id,
                    result.parent_step_id,
                )

                if current_step is not None and current_step.status is AgentRunStepStatus.RUNNING:
                    now = datetime.now(UTC)
                    steps_repository.transition(
                        request.parent_run_id,
                        result.parent_step_id,
                        status=AgentRunStepStatus.FAILED,
                        updated_at=now,
                        completed_at=now,
                        error=(
                            failed_child.error_message if failed_child is not None else str(exc)
                        ),
                        failure_category=(
                            failed_child.error_type
                            if failed_child is not None
                            else type(exc).__name__
                        ),
                    )
            finally:
                steps_repository.close()

            raise

        completed_child = self._agent_run_repository.get(child.run_id)

        if completed_child is None:
            raise RuntimeError(f"delegated child run disappeared: {child.run_id}")

        self._reconcile_completed_child(
            request.parent_run_id,
            result.parent_step_id,
            completed_child,
        )

        return AgentDelegationResult(
            child_run=completed_child,
            parent_step_id=result.parent_step_id,
            created=result.created,
        )

    def reconcile_child_run(
        self,
        *,
        parent_run_id: str,
        parent_step_id: str,
        child_run: AgentRun,
    ) -> None:
        """Reconcile a terminal delegated child into its parent step.

        This method is intentionally public so lifecycle services that can
        complete a child outside AgentDelegationService.execute_delegation()
        (for example approval continuation) can finalize the durable
        delegation step without re-executing the child.
        """
        self._reconcile_completed_child(
            parent_run_id,
            parent_step_id,
            child_run,
        )

    def _reconcile_completed_child(
        self,
        parent_run_id: str,
        parent_step_id: str,
        child: AgentRun,
    ) -> None:
        if child.status not in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.REJECTED,
        }:
            raise RuntimeError(
                "delegated child execution returned a non-terminal run: " f"{child.status.value}"
            )

        steps_repository = self._agent_run_steps_repository_factory()
        try:
            current_step = steps_repository.get(
                parent_run_id,
                parent_step_id,
            )

            if current_step is None:
                raise RuntimeError("delegation parent step disappeared while reconciling child.")

            if current_step.status is AgentRunStepStatus.COMPLETED:
                if child.status is AgentRunStatus.COMPLETED:
                    return

                raise RuntimeError(
                    "delegation parent step is COMPLETED while child run is "
                    f"{child.status.value}."
                )

            if current_step.status is AgentRunStepStatus.FAILED:
                if child.status in {
                    AgentRunStatus.FAILED,
                    AgentRunStatus.REJECTED,
                }:
                    return

                raise RuntimeError(
                    "delegation parent step is FAILED while child run is " f"{child.status.value}."
                )

            if current_step.status is not AgentRunStepStatus.RUNNING:
                raise RuntimeError(
                    "delegation parent step is not RUNNING while reconciling "
                    f"terminal child: {current_step.status.value}"
                )

            now = datetime.now(UTC)

            if child.status is AgentRunStatus.COMPLETED:
                updated_step = steps_repository.transition(
                    parent_run_id,
                    parent_step_id,
                    status=AgentRunStepStatus.COMPLETED,
                    updated_at=now,
                    completed_at=now,
                    output=child.output,
                )
            else:
                updated_step = steps_repository.transition(
                    parent_run_id,
                    parent_step_id,
                    status=AgentRunStepStatus.FAILED,
                    updated_at=now,
                    completed_at=now,
                    error=child.error_message,
                    failure_category=child.error_type,
                )

            if updated_step is None:
                raise RuntimeError("delegation parent step disappeared while reconciling child.")
        finally:
            steps_repository.close()

    def _delegation_depth(self, parent: AgentRun) -> int:
        """
        Calculate hierarchy depth from durable parent relationships.

        Root = 0
        Child of root = 1
        Grandchild = 2
        ...
        """

        depth = 0
        current = parent
        visited: set[str] = set()

        while current.parent_run_id is not None:
            if current.run_id in visited:
                raise RuntimeError("agent run hierarchy contains a parent cycle.")

            visited.add(current.run_id)

            ancestor = self._agent_run_repository.get(current.parent_run_id)

            if ancestor is None:
                raise RuntimeError(
                    "agent run hierarchy references a missing parent run: "
                    f"{current.parent_run_id}"
                )

            if ancestor.root_run_id != parent.root_run_id:
                raise RuntimeError("agent run hierarchy contains inconsistent root_run_id.")

            depth += 1
            current = ancestor

        if current.root_run_id != parent.root_run_id:
            raise RuntimeError("agent run hierarchy has an inconsistent root run.")

        return depth

    @staticmethod
    def _inherit_identity(
        parent: AgentRun,
        child_request: AgentRequest,
    ) -> AgentRequest:
        values: dict[str, object] = {}

        for field_name in (
            "user_id",
            "principal",
            "tenant_id",
            "session_id",
        ):
            parent_value = getattr(parent, field_name)
            child_value = getattr(child_request, field_name)

            if parent_value is not None:
                if child_value is not None and child_value != parent_value:
                    raise ValueError(
                        "child agent identity must inherit the parent " f"{field_name}."
                    )

                values[field_name] = parent_value
            else:
                values[field_name] = child_value

        return replace(child_request, **values)

    @staticmethod
    def _normalize_idempotency_key(
        idempotency_key: str | None,
    ) -> str | None:
        if idempotency_key is None:
            return None

        normalized = idempotency_key.strip()

        if not normalized:
            raise ValueError("idempotency_key must not be empty when provided.")

        return normalized

    @staticmethod
    def _build_idempotency_key(
        *,
        parent_run_id: str,
        idempotency_key: str | None,
    ) -> str | None:
        if idempotency_key is None:
            return None

        return f"deldai:delegation:{parent_run_id}:{idempotency_key}"

    @staticmethod
    def _build_step_id(
        *,
        parent_run_id: str,
        idempotency_key: str | None,
    ) -> str:
        if idempotency_key is not None:
            return str(
                uuid5(
                    NAMESPACE_URL,
                    f"deldai:delegation-step:{parent_run_id}:" f"{idempotency_key}",
                )
            )

        return str(uuid4())

    @staticmethod
    def _build_child_run_id(
        *,
        parent_run_id: str,
        parent_step_id: str,
        idempotency_key: str | None,
    ) -> str:
        if idempotency_key is not None:
            return str(
                uuid5(
                    NAMESPACE_URL,
                    f"deldai:child-run:{parent_run_id}:" f"{idempotency_key}",
                )
            )

        return str(uuid4())

    @staticmethod
    def _validate_existing_child(
        existing: AgentRun,
        *,
        parent: AgentRun,
        child_agent_name: str,
    ) -> None:
        if existing.agent_name != child_agent_name:
            raise ValueError(
                "delegation idempotency key is already associated " "with a different child agent."
            )

        if existing.parent_run_id != parent.run_id:
            raise ValueError(
                "delegation idempotency key is already associated " "with a different parent run."
            )

        if existing.root_run_id != parent.root_run_id:
            raise ValueError(
                "delegation idempotency key is associated with an " "inconsistent root run."
            )

        if existing.parent_step_id is None:
            raise RuntimeError("existing delegated child is missing parent_step_id.")
