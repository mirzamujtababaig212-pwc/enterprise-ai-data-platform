from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SnowflakeColumnMetadata:
    """Canonical DELDAI metadata for a Snowflake table column."""

    name: str | None = None
    type_name: str | None = None
    type_text: str | None = None
    nullable: bool | None = None
    comment: str | None = None
    position: int | None = None


@dataclass(frozen=True)
class SnowflakeTableMetadata:
    """Canonical DELDAI metadata for a Snowflake table."""

    full_name: str | None = None
    database: str | None = None
    schema: str | None = None
    name: str | None = None
    table_type: str | None = None
    owner: str | None = None
    comment: str | None = None
    columns: tuple[SnowflakeColumnMetadata, ...] = ()
