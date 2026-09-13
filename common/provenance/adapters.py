from __future__ import annotations

from common.databricks.metadata import DatabricksTableMetadata
from common.fabric.metadata import FabricTableMetadata
from common.provenance.source import EnterpriseSourceRef
from common.snowflake.metadata import SnowflakeTableMetadata


def databricks_source_ref(
    metadata: DatabricksTableMetadata,
) -> EnterpriseSourceRef:
    """Project Databricks table metadata into the canonical source identity."""
    namespace_parts = [part for part in (metadata.catalog, metadata.schema) if part is not None]

    return EnterpriseSourceRef(
        platform="databricks",
        object_type="table",
        object_name=metadata.name or metadata.full_name,
        object_id=metadata.table_id,
        namespace=".".join(namespace_parts) or None,
    )


def snowflake_source_ref(
    metadata: SnowflakeTableMetadata,
) -> EnterpriseSourceRef:
    """Project Snowflake table metadata into the canonical source identity."""
    namespace_parts = [part for part in (metadata.database, metadata.schema) if part is not None]

    return EnterpriseSourceRef(
        platform="snowflake",
        object_type="table",
        object_name=metadata.name or metadata.full_name,
        namespace=".".join(namespace_parts) or None,
    )


def fabric_source_ref(
    metadata: FabricTableMetadata,
) -> EnterpriseSourceRef:
    """Project Fabric table metadata into the canonical source identity."""
    namespace_parts = [
        part
        for part in (
            metadata.workspace_name,
            metadata.lakehouse_name,
            metadata.schema_name,
        )
        if part is not None
    ]

    return EnterpriseSourceRef(
        platform="fabric",
        object_type="table",
        object_name=metadata.table_name,
        object_id=metadata.table_id,
        namespace=".".join(namespace_parts) or None,
    )
