from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DbtColumn:
    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DbtModel:
    unique_id: str
    name: str
    resource_type: str
    database: str | None = None
    schema: str | None = None
    alias: str | None = None
    description: str = ""
    materialization: str | None = None
    tags: tuple[str, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    columns: tuple[DbtColumn, ...] = ()
