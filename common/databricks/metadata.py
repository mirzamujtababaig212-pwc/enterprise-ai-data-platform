from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatabricksColumnMetadata:
    """Canonical DELDAI metadata for a Databricks table column."""

    name: str | None = None
    type_name: str | None = None
    type_text: str | None = None
    nullable: bool | None = None
    comment: str | None = None
    position: int | None = None
    partition_index: int | None = None


@dataclass(frozen=True)
class DatabricksTableMetadata:
    """Canonical DELDAI metadata for a Databricks table."""

    full_name: str | None = None
    catalog: str | None = None
    schema: str | None = None
    name: str | None = None
    table_type: str | None = None
    owner: str | None = None
    comment: str | None = None
    table_id: str | None = None
    properties: dict[str, str] = field(default_factory=dict)
    columns: tuple[DatabricksColumnMetadata, ...] = ()
