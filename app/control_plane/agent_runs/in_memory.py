from __future__ import annotations

from datetime import datetime

from threading import Lock

from app.control_plane.agent_runs.exceptions import (
    AgentRunNotFoundError,
    DuplicateAgentRunError,
)
from app.control_plane.agent_runs.models import AgentRun, AgentRunStatus


class InMemoryAgentRunRepository:
    def __init__(self) -> None:
        self._runs: dict[str, AgentRun] = {}
        self._lock = Lock()

    def create(
        self,
        run: AgentRun,
        *,
        commit: bool = True,
    ) -> AgentRun:
        del commit

        with self._lock:
            if run.run_id in self._runs:
                raise DuplicateAgentRunError(f"agent run already exists: {run.run_id}")

            self._runs[run.run_id] = run

        return run

    def get(self, run_id: str) -> AgentRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def get_for_tenant(
        self,
        run_id: str,
        tenant_id: str,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if run is None or run.tenant_id != tenant_id:
                return None

            return run

    def get_by_idempotency_key(
        self,
        tenant_id: str,
        user_id: str,
        idempotency_key: str,
    ) -> AgentRun | None:
        with self._lock:
            for run in self._runs.values():
                if (
                    run.tenant_id == tenant_id
                    and run.user_id == user_id
                    and run.idempotency_key == idempotency_key
                ):
                    return run

        return None

    def update(
        self,
        run: AgentRun,
        *,
        commit: bool = True,
    ) -> AgentRun:
        del commit

        with self._lock:
            if run.run_id not in self._runs:
                raise AgentRunNotFoundError(f"agent run not found: {run.run_id}")

            self._runs[run.run_id] = run

        return run

    def claim_for_recovery(
        self,
        run_id: str,
        *,
        started_at: datetime,
        lease_id: str,
        lease_expires_at: datetime,
        max_recovery_attempts: int,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status != AgentRunStatus.FAILED
                or run.cancellation_requested
                or run.recovery_attempts >= max_recovery_attempts
            ):
                return None

            claimed = run.transition_to(AgentRunStatus.RUNNING).model_copy(
                update={
                    "completed_at": None,
                    "error_type": None,
                    "error_message": None,
                    "output": None,
                    "lease_id": lease_id,
                    "lease_expires_at": lease_expires_at,
                    "recovery_attempts": run.recovery_attempts + 1,
                }
            )

            self._runs[run_id] = claimed
            return claimed

    def claim_pending_run(
        self,
        run_id: str,
        *,
        started_at: datetime,
        lease_id: str,
        lease_expires_at: datetime,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.PENDING
                or run.cancellation_requested
            ):
                return None

            claimed = run.transition_to(AgentRunStatus.RUNNING).model_copy(
                update={
                    "started_at": started_at,
                    "lease_id": lease_id,
                    "lease_expires_at": lease_expires_at,
                }
            )

            self._runs[run_id] = claimed
            return claimed

    def claim_waiting_for_approval(
        self,
        run_id: str,
        *,
        lease_id: str,
        lease_expires_at: datetime,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.WAITING_FOR_APPROVAL
                or run.cancellation_requested
            ):
                return None

            claimed = run.transition_to(AgentRunStatus.RUNNING).model_copy(
                update={
                    "lease_id": lease_id,
                    "lease_expires_at": lease_expires_at,
                }
            )

            self._runs[run_id] = claimed
            return claimed

    def reject_waiting_for_approval(
        self,
        run_id: str,
        *,
        completed_at: datetime,
        error_type: str,
        error_message: str,
        commit: bool = True,
    ) -> AgentRun | None:
        del commit

        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.WAITING_FOR_APPROVAL
                or run.cancellation_requested
            ):
                return None

            rejected = run.transition_to(AgentRunStatus.REJECTED).model_copy(
                update={
                    "completed_at": completed_at,
                    "error_type": error_type,
                    "error_message": error_message,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            )

            self._runs[run_id] = rejected
            return rejected

    def heartbeat(
        self,
        run_id: str,
        *,
        lease_id: str,
        lease_expires_at: datetime,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.RUNNING
                or run.lease_id != lease_id
                or run.lease_expires_at is None
            ):
                return None

            updated = run.model_copy(
                update={
                    "lease_expires_at": lease_expires_at,
                }
            )

            self._runs[run_id] = updated
            return updated

    def claim_expired_running_run(
        self,
        run_id: str,
        *,
        stale_before: datetime,
        started_at: datetime,
        lease_id: str,
        lease_expires_at: datetime,
        max_recovery_attempts: int,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.RUNNING
                or run.cancellation_requested
                or run.lease_expires_at is None
                or run.lease_expires_at >= stale_before
                or run.recovery_attempts >= max_recovery_attempts
            ):
                return None

            claimed = run.model_copy(
                update={
                    "completed_at": None,
                    "error_type": None,
                    "error_message": None,
                    "output": None,
                    "lease_id": lease_id,
                    "lease_expires_at": lease_expires_at,
                    "recovery_attempts": run.recovery_attempts + 1,
                }
            )

            self._runs[run_id] = claimed
            return claimed

    def fail_recovery_exhausted(
        self,
        run_id: str,
        *,
        completed_at: datetime,
        max_recovery_attempts: int,
        error_type: str,
        error_message: str,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.RUNNING
                or run.cancellation_requested
                or run.lease_expires_at is None
                or run.recovery_attempts < max_recovery_attempts
                or run.lease_expires_at >= completed_at
            ):
                return None

            failed = run.model_copy(
                update={
                    "status": AgentRunStatus.FAILED,
                    "completed_at": completed_at,
                    "error_type": error_type,
                    "error_message": error_message,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            )

            self._runs[run_id] = failed
            return failed

    def list_expired_running_runs(
        self,
        *,
        stale_before: datetime,
        limit: int = 100,
    ) -> list[AgentRun]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        with self._lock:
            runs = [
                run
                for run in self._runs.values()
                if (
                    run.status is AgentRunStatus.RUNNING
                    and not run.cancellation_requested
                    and run.lease_expires_at is not None
                    and run.lease_expires_at < stale_before
                )
            ]

        return sorted(
            runs,
            key=lambda run: (run.lease_expires_at, run.run_id),
        )[:limit]

    def transition_to_waiting_for_approval_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        updated_at: datetime,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.RUNNING
                or run.lease_id != lease_id
                or run.lease_expires_at is None
                or run.lease_expires_at <= updated_at
            ):
                return None

            waiting = run.transition_to(AgentRunStatus.WAITING_FOR_APPROVAL).model_copy(
                update={
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            )

            self._runs[run_id] = waiting
            return waiting

    def complete_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
        output,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.RUNNING
                or run.lease_id != lease_id
                or run.lease_expires_at is None
                or run.lease_expires_at <= completed_at
            ):
                return None

            completed = run.transition_to(AgentRunStatus.COMPLETED).model_copy(
                update={
                    "completed_at": completed_at,
                    "output": output,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            )

            self._runs[run_id] = completed
            return completed

    def fail_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
        error_type: str,
        error_message: str,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.RUNNING
                or run.lease_id != lease_id
                or run.lease_expires_at is None
                or run.lease_expires_at <= completed_at
            ):
                return None

            failed = run.transition_to(AgentRunStatus.FAILED).model_copy(
                update={
                    "completed_at": completed_at,
                    "error_type": error_type,
                    "error_message": error_message,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            )

            self._runs[run_id] = failed
            return failed

    def cancel_if_owner(
        self,
        run_id: str,
        *,
        lease_id: str,
        completed_at: datetime,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if (
                run is None
                or run.status is not AgentRunStatus.RUNNING
                or run.lease_id != lease_id
                or run.lease_expires_at is None
                or run.lease_expires_at <= completed_at
            ):
                return None

            cancelled = run.transition_to(AgentRunStatus.CANCELLED).model_copy(
                update={
                    "completed_at": completed_at,
                    "lease_id": None,
                    "lease_expires_at": None,
                }
            )

            self._runs[run_id] = cancelled
            return cancelled

    def list(
        self,
        *,
        tenant_id: str | None = None,
        agent_name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
        status: AgentRunStatus | None = None,
        limit: int = 100,
    ) -> list[AgentRun]:
        with self._lock:
            runs = list(self._runs.values())

        if tenant_id is not None:
            runs = [run for run in runs if run.tenant_id == tenant_id]

        if agent_name is not None:
            runs = [run for run in runs if run.agent_name == agent_name]

        if session_id is not None:
            runs = [run for run in runs if run.session_id == session_id]

        if user_id is not None:
            runs = [run for run in runs if run.user_id == user_id]

        if status is not None:
            runs = [run for run in runs if run.status == status]

        return runs[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._runs.clear()

    def request_cancellation(
        self,
        run_id: str,
        *,
        requested_at: datetime,
    ) -> AgentRun | None:
        with self._lock:
            run = self._runs.get(run_id)

            if run is None:
                return None

            if run.status is not AgentRunStatus.RUNNING:
                return None

            if run.cancellation_requested:
                return run

            requested = run.model_copy(
                update={
                    "cancellation_requested": True,
                    "cancellation_requested_at": requested_at,
                }
            )

            self._runs[run_id] = requested

            return requested
