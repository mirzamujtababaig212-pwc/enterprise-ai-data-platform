from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

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
    claim_token: str | None = None


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
        *,
        claim_token: str,
    ) -> None:
        raise NotImplementedError

    async def release(
        self,
        key: ToolExecutionIdempotencyKey,
        *,
        claim_token: str,
    ) -> None:
        raise NotImplementedError

    async def mark_ambiguous(
        self,
        key: ToolExecutionIdempotencyKey,
        *,
        claim_token: str,
    ) -> None:
        raise NotImplementedError


class InMemoryToolExecutionIdempotencyStore(
    ToolExecutionIdempotencyStore,
):
    def __init__(self) -> None:
        self._completed: dict[
            ToolExecutionIdempotencyKey,
            ToolExecutionResult,
        ] = {}
        self._in_progress: dict[
            ToolExecutionIdempotencyKey,
            str,
        ] = {}
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

            claim_token = str(uuid4())
            self._in_progress[key] = claim_token

            return ToolIdempotencyClaim(
                status=ToolIdempotencyClaimStatus.CLAIMED,
                claim_token=claim_token,
            )

    async def complete(
        self,
        key: ToolExecutionIdempotencyKey,
        result: ToolExecutionResult,
        *,
        claim_token: str,
    ) -> None:
        async with self._lock:
            current_token = self._in_progress.get(key)

            if current_token != claim_token:
                raise RuntimeError(
                    "Unable to complete idempotency record because the "
                    "claim is no longer owned by the caller."
                )

            self._in_progress.pop(key)
            self._completed[key] = result

    async def release(
        self,
        key: ToolExecutionIdempotencyKey,
        *,
        claim_token: str,
    ) -> None:
        async with self._lock:
            current_token = self._in_progress.get(key)

            if current_token != claim_token:
                raise RuntimeError(
                    "Unable to release idempotency record because the "
                    "claim is no longer owned by the caller."
                )

            self._in_progress.pop(key)

    async def mark_ambiguous(
        self,
        key: ToolExecutionIdempotencyKey,
        *,
        claim_token: str,
    ) -> None:
        async with self._lock:
            current_token = self._in_progress.get(key)

            if current_token != claim_token:
                raise RuntimeError(
                    "Unable to mark idempotency record ambiguous because "
                    "the claim is no longer owned by the caller."
                )

            self._in_progress.pop(key)
            self._ambiguous.add(key)
