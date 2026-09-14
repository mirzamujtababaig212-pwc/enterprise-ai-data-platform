from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AwsGlueColumnMetadata:
    """Canonical DELDAI metadata for an AWS Glue table column."""

    name: str | None = None
    type_name: str | None = None
    comment: str | None = None
    position: int | None = None
    partition_index: int | None = None


@dataclass(frozen=True)
class AwsGlueTableMetadata:
    """Canonical DELDAI metadata for an AWS Glue Data Catalog table."""

    catalog_id: str | None = None
    database_name: str | None = None
    name: str | None = None
    table_type: str | None = None
    owner: str | None = None
    description: str | None = None
    location: str | None = None
    input_format: str | None = None
    output_format: str | None = None
    serde_library: str | None = None
    parameters: dict[str, str] = field(default_factory=dict)
    columns: tuple[AwsGlueColumnMetadata, ...] = ()
