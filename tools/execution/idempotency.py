from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum

from tools.models import ToolExecutionResult


def build_external_idempotency_key(
    run_id: str,
    call_id: str,
    tool_name: str,
) -> str:
    key = ToolExecutionIdempotencyKey(
        run_id=run_id,
        call_id=call_id,
        tool_name=tool_name,
    )

    return f"deldai:{key.run_id}:{key.call_id}:{key.tool_name}"


@dataclass(frozen=True)
class ToolExecutionIdempotencyKey:
    run_id: str
    call_id: str
    tool_name: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("run_id", self.run_id),
            ("call_id", self.call_id),
            ("tool_name", self.tool_name),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")


class ToolIdempotencyClaimStatus(StrEnum):
    CLAIMED = "claimed"
    COMPLETED = "completed"
    IN_PROGRESS = "in_progress"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ToolIdempotencyClaim:
    status: ToolIdempotencyClaimStatus
    result: ToolExecutionResult | None = None


class ToolExecutionIdempotencyStore:
    async def claim(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> ToolIdempotencyClaim:
        raise NotImplementedError

    async def complete(
        self,
        key: ToolExecutionIdempotencyKey,
        result: ToolExecutionResult,
    ) -> None:
        raise NotImplementedError

    async def release(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> None:
        raise NotImplementedError

    async def mark_ambiguous(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> None:
        raise NotImplementedError


class InMemoryToolExecutionIdempotencyStore(ToolExecutionIdempotencyStore):
    def __init__(self) -> None:
        self._completed: dict[
            ToolExecutionIdempotencyKey,
            ToolExecutionResult,
        ] = {}
        self._in_progress: set[ToolExecutionIdempotencyKey] = set()
        self._ambiguous: set[ToolExecutionIdempotencyKey] = set()
        self._lock = asyncio.Lock()

    async def claim(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> ToolIdempotencyClaim:
        async with self._lock:
            if key in self._ambiguous:
                return ToolIdempotencyClaim(
                    status=ToolIdempotencyClaimStatus.AMBIGUOUS,
                )

            completed = self._completed.get(key)

            if completed is not None:
                return ToolIdempotencyClaim(
                    status=ToolIdempotencyClaimStatus.COMPLETED,
                    result=completed,
                )

            if key in self._in_progress:
                return ToolIdempotencyClaim(
                    status=ToolIdempotencyClaimStatus.IN_PROGRESS,
                )

            self._in_progress.add(key)

            return ToolIdempotencyClaim(
                status=ToolIdempotencyClaimStatus.CLAIMED,
            )

    async def complete(
        self,
        key: ToolExecutionIdempotencyKey,
        result: ToolExecutionResult,
    ) -> None:
        async with self._lock:
            self._in_progress.discard(key)
            self._completed[key] = result

    async def release(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> None:
        async with self._lock:
            self._in_progress.discard(key)

    async def mark_ambiguous(
        self,
        key: ToolExecutionIdempotencyKey,
    ) -> None:
        async with self._lock:
            if key not in self._in_progress:
                raise RuntimeError("Cannot mark tool execution ambiguous without an active claim.")

            self._in_progress.discard(key)
            self._ambiguous.add(key)
