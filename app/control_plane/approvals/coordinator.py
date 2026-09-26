from __future__ import annotations

from collections.abc import Callable
from uuid import NAMESPACE_URL, uuid5

from tools.execution.approval import (
    ToolApprovalCoordinator,
    ToolApprovalDecision,
    ToolApprovalDisposition,
    ToolApprovalRequest,
)
from tools.execution.idempotency import build_external_idempotency_key

from .models import ApprovalRequest, ApprovalStatus
from .policy import ApprovalEvaluationRequest, ApprovalPolicy
from .repository import ApprovalRequestRepository


class ControlPlaneApprovalCoordinator(ToolApprovalCoordinator):
    """
    Coordinates approval policy evaluation with durable approval state.

    The coordinator does not execute tools and does not perform authorization.
    It only determines whether an already-authorized tool execution may proceed.
    """

    def __init__(
        self,
        *,
        policy: ApprovalPolicy,
        repository_factory: Callable[[], ApprovalRequestRepository],
    ) -> None:
        self._policy = policy
        self._repository_factory = repository_factory

    async def evaluate(
        self,
        request: ToolApprovalRequest,
    ) -> ToolApprovalDecision:
        policy_request = ApprovalEvaluationRequest(
            tool_name=request.tool_name,
            tool_metadata=request.tool_metadata,
            arguments=request.arguments,
            run_id=request.run_id,
            step_id=request.step_id,
            call_id=request.call_id,
            agent_name=request.agent_name,
            session_id=request.session_id,
            user_id=request.user_id,
            principal=request.principal,
            tenant_id=request.tenant_id,
        )

        policy_decision = self._policy.evaluate(policy_request)

        if not policy_decision.approval_required:
            return ToolApprovalDecision(
                disposition=ToolApprovalDisposition.NOT_REQUIRED,
                policy_name=policy_decision.policy_name,
                policy_version=policy_decision.policy_version,
                risk_tier=policy_decision.risk_tier,
                requested_action=policy_decision.requested_action,
                policy_metadata=policy_decision.policy_metadata,
            )

        approval_id = self._build_approval_id(
            run_id=request.run_id,
            step_id=request.step_id,
            call_id=request.call_id,
        )
        idempotency_key = build_external_idempotency_key(
            request.run_id,
            request.call_id,
            request.tool_name,
        )

        repository = self._repository_factory()

        try:
            existing = repository.get(approval_id)

            if existing is None:
                approval = ApprovalRequest(
                    approval_id=approval_id,
                    run_id=request.run_id,
                    step_id=request.step_id,
                    call_id=request.call_id,
                    tool_name=request.tool_name,
                    idempotency_key=idempotency_key,
                    status=ApprovalStatus.PENDING,
                    policy_name=policy_decision.policy_name,
                    policy_version=policy_decision.policy_version,
                    risk_tier=policy_decision.risk_tier,
                    requested_action=policy_decision.requested_action,
                    policy_metadata=dict(policy_decision.policy_metadata),
                )
                repository.create(approval, commit=True)
                status = ApprovalStatus.PENDING
            else:
                status = existing.status

            disposition = {
                ApprovalStatus.PENDING: ToolApprovalDisposition.PENDING,
                ApprovalStatus.APPROVED: ToolApprovalDisposition.APPROVED,
                ApprovalStatus.REJECTED: ToolApprovalDisposition.REJECTED,
            }[status]

            return ToolApprovalDecision(
                disposition=disposition,
                approval_id=approval_id,
                policy_name=policy_decision.policy_name,
                policy_version=policy_decision.policy_version,
                risk_tier=policy_decision.risk_tier,
                requested_action=policy_decision.requested_action,
                policy_metadata=policy_decision.policy_metadata,
            )
        finally:
            close = getattr(repository, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _build_approval_id(
        *,
        run_id: str,
        step_id: str,
        call_id: str,
    ) -> str:
        return str(
            uuid5(
                NAMESPACE_URL,
                f"deldai:approval:{run_id}:{step_id}:{call_id}",
            )
        )
