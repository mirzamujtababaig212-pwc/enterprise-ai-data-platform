from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FabricColumnMetadata:
    name: str | None = None
    type_name: str | None = None
    type_text: str | None = None
    nullable: bool | None = None
    comment: str | None = None
    position: int | None = None
    partition_index: int | None = None


@dataclass(frozen=True)
class FabricTableMetadata:
    workspace_id: str | None = None
    workspace_name: str | None = None
    lakehouse_id: str | None = None
    lakehouse_name: str | None = None
    catalog_name: str | None = None
    schema_name: str | None = None
    table_id: str | None = None
    table_name: str | None = None
    table_type: str | None = None
    location: str | None = None
    format: str | None = None
    owner: str | None = None
    comment: str | None = None
    properties: dict[str, str] = field(default_factory=dict)
    columns: tuple[FabricColumnMetadata, ...] = ()
