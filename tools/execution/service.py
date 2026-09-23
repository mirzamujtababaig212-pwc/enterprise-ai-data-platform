from __future__ import annotations

import asyncio
import logging
from typing import Any

from jsonschema import SchemaError, ValidationError, validate

from tools.authorization.audit import (
    ToolAuthorizationAuditRecord,
    ToolAuthorizationAuditSink,
)
from tools.authorization.service import ToolAuthorizationService
from tools.contracts import ToolRegistry
from ai_platform.agents.policy import (
    PolicyViolationError,
    TenantPolicyEngine,
)
from tools.execution.context import ToolExecutionContext
from tools.execution.exceptions import ToolExecutionOwnershipLostError
from tools.execution.idempotency import (
    ToolExecutionIdempotencyKey,
    ToolExecutionIdempotencyStore,
    ToolIdempotencyClaimStatus,
)
from tools.models import (
    ToolExecutionFailureCategory,
    ToolExecutionResult,
)

logger = logging.getLogger(__name__)


class ToolExecutionService:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        authorization_service: ToolAuthorizationService | None = None,
        audit_sink: ToolAuthorizationAuditSink | None = None,
        idempotency_store: ToolExecutionIdempotencyStore | None = None,
        tenant_policy_engine: TenantPolicyEngine | None = None,
        default_timeout_seconds: float = 30.0,
    ):
        if default_timeout_seconds <= 0:
            raise ValueError("default_timeout_seconds must be greater than zero.")

        self.registry = registry
        self.authorization_service = authorization_service
        self.audit_sink = audit_sink
        self.idempotency_store = idempotency_store
        self.tenant_policy_engine = tenant_policy_engine
        self.default_timeout_seconds = default_timeout_seconds

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        principal: str | None = None,
        timeout_seconds: float | None = None,
        execution_context: ToolExecutionContext | dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        execution_context = self._normalize_execution_context(execution_context)

        if not tool_name.strip():
            raise ValueError("Tool name must not be empty.")

        if arguments is None:
            raise ValueError("Tool arguments must not be None.")

        timeout = self.default_timeout_seconds if timeout_seconds is None else timeout_seconds

        if timeout <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        tool = await self.registry.get(tool_name)

        if tool is None:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool not found: {tool_name}",
                failure_category=ToolExecutionFailureCategory.TOOL_NOT_FOUND,
            )

        if not tool.definition.enabled:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool is disabled: {tool_name}",
                failure_category=ToolExecutionFailureCategory.TOOL_DISABLED,
            )

        try:
            validate(
                instance=arguments,
                schema=tool.definition.input_schema,
            )
        except ValidationError as exc:
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool arguments failed schema validation: {exc.message}",
                failure_category=ToolExecutionFailureCategory.SCHEMA_VALIDATION,
            )
        except SchemaError as exc:
            logger.exception(
                "Tool has an invalid input schema: tool_name=%s",
                tool_name,
            )
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                error=f"Tool input schema is invalid: {exc.message}",
                failure_category=ToolExecutionFailureCategory.INVALID_SCHEMA,
            )

        if self.tenant_policy_engine is not None:
            tenant_id = execution_context.tenant_id if execution_context is not None else None

            if tenant_id is None or not tenant_id.strip():
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error="Tenant identity is required when tenant policy enforcement is enabled.",
                    failure_category=ToolExecutionFailureCategory.TENANT_POLICY,
                )

            server_id = tool.definition.metadata.get("mcp_server")

            try:
                self.tenant_policy_engine.validate_tool_execution(
                    tenant_id=tenant_id,
                    tool_name=tool_name,
                    server_id=server_id,
                )
            except PolicyViolationError as exc:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(exc),
                    failure_category=ToolExecutionFailureCategory.TENANT_POLICY,
                )

        if self.authorization_service is not None:
            if principal is None or not principal.strip():
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=("Principal is required when " "tool authorization is enabled."),
                    failure_category=ToolExecutionFailureCategory.MISSING_PRINCIPAL,
                )

            authorization = await self.authorization_service.authorize(
                principal,
                tool_name,
                metadata=tool.definition.metadata,
            )

            await self._audit_authorization(
                principal=principal,
                tool_name=tool_name,
                allowed=authorization.allowed,
                reason=authorization.reason,
                policy_id=authorization.policy_id,
                policy_version=authorization.policy_version,
                execution_context=execution_context,
            )

            if not authorization.allowed:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=(authorization.reason or "Tool execution is not authorized."),
                    failure_category=ToolExecutionFailureCategory.AUTHORIZATION,
                )

        idempotency_key = None
        claim_token = None
        ownership_outcome_ambiguous = False

        if (
            self.idempotency_store is not None
            and execution_context is not None
            and execution_context.run_id is not None
            and execution_context.call_id is not None
        ):
            idempotency_key = ToolExecutionIdempotencyKey(
                run_id=execution_context.run_id,
                call_id=execution_context.call_id,
                tool_name=tool_name,
            )

            claim = await self.idempotency_store.claim(idempotency_key)

            if claim.status == ToolIdempotencyClaimStatus.CLAIMED:
                claim_token = claim.claim_token
                if claim_token is None:
                    raise RuntimeError(
                        "Idempotency store returned a claimed state without " "a claim token."
                    )

            if claim.status == ToolIdempotencyClaimStatus.AMBIGUOUS:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=(
                        "Tool execution has an ambiguous external outcome and "
                        "must not be retried automatically."
                    ),
                    failure_category=ToolExecutionFailureCategory.EXECUTION_AMBIGUOUS,
                )

            if claim.status == ToolIdempotencyClaimStatus.COMPLETED:
                if claim.result is None:
                    raise RuntimeError(
                        "Idempotency store returned a completed claim without a result."
                    )
                return claim.result

            if claim.status == ToolIdempotencyClaimStatus.IN_PROGRESS:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    error=(
                        "Tool execution is already in progress for "
                        f"run_id={execution_context.run_id}, "
                        f"call_id={execution_context.call_id}."
                    ),
                    failure_category=ToolExecutionFailureCategory.EXECUTION_IN_PROGRESS,
                )

        policy = tool.definition.execution_policy

        try:
            for attempt in range(policy.max_retries + 1):
                try:
                    contextual_execute = getattr(tool, "execute_with_context", None)

                    self._raise_if_execution_ownership_lost(execution_context)

                    if contextual_execute is not None:
                        execution = contextual_execute(
                            arguments,
                            execution_context,
                        )
                    else:
                        execution = tool.execute(arguments)

                    output = await asyncio.wait_for(
                        execution,
                        timeout=timeout,
                    )

                    if (
                        execution_context is not None
                        and execution_context.execution_ownership_lost is not None
                        and execution_context.execution_ownership_lost.is_set()
                    ):
                        ownership_outcome_ambiguous = True

                        if idempotency_key is not None:
                            await self.idempotency_store.mark_ambiguous(
                                idempotency_key,
                                claim_token=claim_token,
                            )

                        raise ToolExecutionOwnershipLostError(
                            "Tool execution completed after durable run ownership was lost."
                        )

                    result = ToolExecutionResult(
                        tool_name=tool_name,
                        success=True,
                        output=output,
                    )

                    if idempotency_key is not None:
                        await self.idempotency_store.complete(
                            idempotency_key,
                            result,
                            claim_token=claim_token,
                        )

                    return result

                except ToolExecutionOwnershipLostError:
                    raise

                except asyncio.TimeoutError:
                    if (
                        execution_context is not None
                        and execution_context.execution_ownership_lost is not None
                        and execution_context.execution_ownership_lost.is_set()
                    ):
                        ownership_outcome_ambiguous = True

                        if idempotency_key is not None:
                            await self.idempotency_store.mark_ambiguous(
                                idempotency_key,
                                claim_token=claim_token,
                            )

                        raise ToolExecutionOwnershipLostError(
                            "Tool execution lost durable run ownership during tool execution."
                        )

                    result = ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        error=(
                            f"Tool execution timed out after " f"{timeout} seconds: {tool_name}"
                        ),
                        failure_category=ToolExecutionFailureCategory.TIMEOUT,
                    )

                except Exception as exc:
                    result = ToolExecutionResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"{type(exc).__name__}: {exc}",
                        failure_category=ToolExecutionFailureCategory.EXECUTION_ERROR,
                    )

                should_retry = (
                    result.failure_category is not None
                    and result.failure_category in policy.retryable_failure_categories
                    and attempt < policy.max_retries
                )

                if not should_retry:
                    if idempotency_key is not None:
                        await self.idempotency_store.release(
                            idempotency_key,
                            claim_token=claim_token,
                        )
                    return result

                if policy.backoff_seconds > 0:
                    await asyncio.sleep(policy.backoff_seconds)

            raise RuntimeError("Tool execution retry loop exited unexpectedly.")
        except BaseException:
            if idempotency_key is not None and not ownership_outcome_ambiguous:
                await self.idempotency_store.release(
                    idempotency_key,
                    claim_token=claim_token,
                )
            raise

    async def _audit_authorization(
        self,
        *,
        principal: str,
        tool_name: str,
        allowed: bool,
        reason: str | None,
        policy_id: str | None,
        policy_version: str | None,
        execution_context: ToolExecutionContext | None = None,
    ) -> None:
        if self.audit_sink is None:
            return

        record = ToolAuthorizationAuditRecord(
            principal=principal,
            tool_name=tool_name,
            allowed=allowed,
            reason=reason,
            policy_id=policy_id,
            policy_version=policy_version,
            run_id=execution_context.run_id if execution_context else None,
            call_id=execution_context.call_id if execution_context else None,
            agent_name=execution_context.agent_name if execution_context else None,
            session_id=execution_context.session_id if execution_context else None,
        )

        try:
            await self.audit_sink.record(record)
        except Exception:
            logger.exception(
                "Failed to record tool authorization audit: "
                "principal=%s tool_name=%s allowed=%s",
                principal,
                tool_name,
                allowed,
            )

    @staticmethod
    def _raise_if_execution_ownership_lost(
        execution_context: ToolExecutionContext | None,
    ) -> None:
        if (
            execution_context is not None
            and execution_context.execution_ownership_lost is not None
            and execution_context.execution_ownership_lost.is_set()
        ):
            raise ToolExecutionOwnershipLostError("Tool execution lost durable run ownership.")

    @staticmethod
    def _normalize_execution_context(
        execution_context: ToolExecutionContext | dict[str, Any] | None = None,
    ) -> ToolExecutionContext | None:
        if execution_context is None:
            return None

        if isinstance(execution_context, ToolExecutionContext):
            return execution_context

        if isinstance(execution_context, dict):
            return ToolExecutionContext(
                run_id=execution_context.get("run_id"),
                call_id=execution_context.get("call_id"),
                agent_name=execution_context.get("agent_name"),
                session_id=execution_context.get("session_id"),
                user_id=execution_context.get("user_id"),
                principal=execution_context.get("principal"),
                tenant_id=execution_context.get("tenant_id"),
                governance_policy=execution_context.get("governance_policy"),
                request_metadata=execution_context.get("request_metadata", {}),
                execution_ownership_lost=execution_context.get("execution_ownership_lost"),
            )

        raise TypeError("execution_context must be a ToolExecutionContext, dict, or None.")
