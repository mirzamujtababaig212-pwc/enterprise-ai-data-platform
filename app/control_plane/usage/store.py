from __future__ import annotations

from threading import Lock

from app.control_plane.usage.models import UsageEvent


class InMemoryUsageStore:
    def __init__(self) -> None:
        self._events: list[UsageEvent] = []
        self._lock = Lock()

    def record(self, event: UsageEvent) -> UsageEvent:
        with self._lock:
            self._events.append(event)

        return event

    def list(
        self,
        *,
        request_id: str | None = None,
        capability: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[UsageEvent]:
        with self._lock:
            events = list(self._events)

        if request_id is not None:
            events = [event for event in events if event.request_id == request_id]

        if capability is not None:
            events = [event for event in events if event.capability == capability]

        if status is not None:
            events = [event for event in events if event.status == status]

        return events[-limit:]

    def get(self, event_id: str) -> UsageEvent | None:
        with self._lock:
            for event in self._events:
                if event.event_id == event_id:
                    return event

        return None

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
