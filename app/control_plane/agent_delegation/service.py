from __future__ import annotations

from dataclasses import replace
from uuid import NAMESPACE_URL, uuid4, uuid5

from ai_platform.agents.contracts import AgentRegistry
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
from app.control_plane.agent_runs.exceptions import DuplicateAgentRunError
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus
from app.control_plane.agent_runs.repository import AgentRunRepository
from app.control_plane.agent_runs.request_snapshot import AgentRunRequestSnapshot


class AgentDelegationService:
    """
    Control-plane primitive for creating durable child-agent delegations.

    This service does not execute the child agent. Child execution is a
    subsequent lifecycle concern handled by the AgentRuntime in Phase 3.
    """

    def __init__(
        self,
        *,
        agent_registry: AgentRegistry,
        agent_run_repository: AgentRunRepository,
        agent_run_steps_repository_factory,
        policy: AgentDelegationPolicy | None = None,
    ) -> None:
        self._agent_registry = agent_registry
        self._agent_run_repository = agent_run_repository
        self._agent_run_steps_repository_factory = agent_run_steps_repository_factory
        self._policy = policy or AgentDelegationPolicy()

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
