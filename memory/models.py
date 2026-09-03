from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


MemoryType = Literal[
    "working",
    "semantic",
    "episodic",
]


@dataclass(frozen=True)
class MemoryItem:
    """
    Canonical representation of a memory.

    A MemoryItem may represent short-lived working state,
    persistent semantic knowledge, or a historical episode.
    """

    id: str
    memory_type: MemoryType
    content: str
    namespace: str
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    expires_at: datetime | None = None
