from __future__ import annotations

import json
from collections.abc import Callable

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.control_plane.persistence.models import ToolExecutionIdempotencyRecord
from tools.execution.idempotency import (
    ToolExecutionIdempotencyKey,
    ToolExecutionIdempotencyStore,
    ToolIdempotencyClaim,
    ToolIdempotencyClaimStatus,
)
from tools.models import ToolExecutionFailureCategory, ToolExecutionResult


class PostgreSQLToolExecutionIdempotencyStore(
    ToolExecutionIdempotencyStore,
):
    """PostgreSQL-backed durable tool execution idempotency store."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
    ) -> None:
        self._session_factory = session_factory

    async def claim(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> ToolIdempotencyClaim:
        session = self._session_factory()

        try:
            record = ToolExecutionIdempotencyRecord(
                run_id=key.run_id,
                call_id=key.call_id,
                tool_name=key.tool_name,
                status=ToolIdempotencyClaimStatus.CLAIMED.value,
                success=False,
            )

            try:
                session.add(record)
                session.flush()
                session.commit()

                return ToolIdempotencyClaim(
                    status=ToolIdempotencyClaimStatus.CLAIMED,
                )

            except IntegrityError:
                session.rollback()

            existing = session.scalar(
                select(ToolExecutionIdempotencyRecord).where(
                    ToolExecutionIdempotencyRecord.run_id == key.run_id,
                    ToolExecutionIdempotencyRecord.call_id == key.call_id,
                    ToolExecutionIdempotencyRecord.tool_name == key.tool_name,
                )
            )

            if existing is None:
                raise RuntimeError(
                    "Idempotency claim conflicted, but the existing record " "could not be loaded."
                )

            if existing.status == ToolIdempotencyClaimStatus.COMPLETED.value:
                return ToolIdempotencyClaim(
                    status=ToolIdempotencyClaimStatus.COMPLETED,
                    result=self._result_from_record(existing),
                )

            if existing.status == ToolIdempotencyClaimStatus.AMBIGUOUS.value:
                return ToolIdempotencyClaim(
                    status=ToolIdempotencyClaimStatus.AMBIGUOUS,
                )

            return ToolIdempotencyClaim(
                status=ToolIdempotencyClaimStatus.IN_PROGRESS,
            )
        finally:
            session.close()

    async def complete(
        self,
        key: ToolExecutionIdempotencyKey,
        result: ToolExecutionResult,
    ) -> None:
        if not result.success:
            await self.release(key)
            return

        try:
            json.dumps(result.output)
        except (TypeError, ValueError):
            await self.release(key)
            return

        session = self._session_factory()

        try:
            statement = (
                update(ToolExecutionIdempotencyRecord)
                .where(
                    ToolExecutionIdempotencyRecord.run_id == key.run_id,
                    ToolExecutionIdempotencyRecord.call_id == key.call_id,
                    ToolExecutionIdempotencyRecord.tool_name == key.tool_name,
                    ToolExecutionIdempotencyRecord.status
                    == ToolIdempotencyClaimStatus.CLAIMED.value,
                )
                .values(
                    status=ToolIdempotencyClaimStatus.COMPLETED.value,
                    success=True,
                    output=result.output,
                    error=result.error,
                    failure_category=(
                        result.failure_category.value
                        if result.failure_category is not None
                        else None
                    ),
                )
            )

            try:
                updated = session.execute(statement)

                if updated.rowcount != 1:
                    session.rollback()
                    raise RuntimeError(
                        "Unable to complete idempotency record because the "
                        "claim is no longer owned by the caller."
                    )

                session.commit()
            except Exception:
                session.rollback()
                raise
        finally:
            session.close()

    async def release(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> None:
        session = self._session_factory()

        try:
            statement = delete(ToolExecutionIdempotencyRecord).where(
                ToolExecutionIdempotencyRecord.run_id == key.run_id,
                ToolExecutionIdempotencyRecord.call_id == key.call_id,
                ToolExecutionIdempotencyRecord.tool_name == key.tool_name,
            )

            try:
                session.execute(statement)
                session.commit()
            except Exception:
                session.rollback()
                raise
        finally:
            session.close()

    async def mark_ambiguous(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> None:
        session = self._session_factory()

        try:
            statement = (
                update(ToolExecutionIdempotencyRecord)
                .where(
                    ToolExecutionIdempotencyRecord.run_id == key.run_id,
                    ToolExecutionIdempotencyRecord.call_id == key.call_id,
                    ToolExecutionIdempotencyRecord.tool_name == key.tool_name,
                    ToolExecutionIdempotencyRecord.status
                    == ToolIdempotencyClaimStatus.CLAIMED.value,
                )
                .values(
                    status=ToolIdempotencyClaimStatus.AMBIGUOUS.value,
                )
            )

            try:
                updated = session.execute(statement)

                if updated.rowcount != 1:
                    session.rollback()
                    raise RuntimeError(
                        "Unable to mark idempotency record ambiguous because "
                        "the claim is no longer owned by the caller."
                    )

                session.commit()
            except Exception:
                session.rollback()
                raise
        finally:
            session.close()

    @staticmethod
    def _result_from_record(
        record: ToolExecutionIdempotencyRecord,
    ) -> ToolExecutionResult:
        failure_category = None

        if record.failure_category is not None:
            failure_category = ToolExecutionFailureCategory(record.failure_category)

        return ToolExecutionResult(
            tool_name=record.tool_name,
            success=record.success,
            output=record.output,
            error=record.error,
            failure_category=failure_category,
        )
