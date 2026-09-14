from __future__ import annotations

from common.databricks.metadata import DatabricksTableMetadata
from common.fabric.metadata import FabricTableMetadata
from common.provenance.source import EnterpriseSourceRef
from common.snowflake.metadata import SnowflakeTableMetadata
from common.aws.metadata import AwsGlueTableMetadata


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


def s3_source_ref(
    path: str,
) -> EnterpriseSourceRef:
    """Project an S3 dataset path into the canonical source identity."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("S3 path cannot be empty.")

    path = path.strip()

    if not path.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")

    uri = path.rstrip("/")

    remainder = uri[len("s3://") :]

    if not remainder or "/" not in remainder:
        raise ValueError("S3 path must include a bucket and object path.")

    bucket, object_path = remainder.split("/", 1)

    if not bucket or not object_path:
        raise ValueError("S3 path must include a bucket and object path.")

    return EnterpriseSourceRef(
        platform="aws",
        object_type="s3_path",
        object_name=uri,
        namespace=f"s3://{bucket}",
    )


def aws_glue_source_ref(
    metadata: AwsGlueTableMetadata,
) -> EnterpriseSourceRef:
    """Project AWS Glue table metadata into the canonical source identity."""
    return EnterpriseSourceRef(
        platform="aws",
        object_type="table",
        object_name=metadata.name,
        namespace=metadata.database_name,
    )
