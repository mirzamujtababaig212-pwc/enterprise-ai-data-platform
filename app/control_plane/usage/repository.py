from __future__ import annotations

from typing import Protocol

from app.control_plane.usage.models import UsageEvent


class UsageRepository(Protocol):
    def record(self, event: UsageEvent) -> UsageEvent: ...

    def list(
        self,
        *,
        request_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[UsageEvent]: ...

    def get(self, event_id: str) -> UsageEvent | None: ...

    def clear(self) -> None: ...
