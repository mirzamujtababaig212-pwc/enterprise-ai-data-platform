from __future__ import annotations

from typing import Any

from pyspark.sql.types import StructType

from common.aws.glue_schema import spark_schema_to_glue_columns


def _normalize_s3_location(location: str) -> str:
    if not location:
        raise ValueError("AWS Glue table location is required.")

    if location.startswith("s3a://"):
        return f"s3://{location[6:]}"

    if location.startswith("s3://"):
        return location

    raise ValueError("AWS Glue table location must use s3:// or s3a://.")


def build_delta_table_input(
    table_name: str,
    schema: StructType,
    location: str,
    description: str | None = None,
) -> dict[str, Any]:
    """Build an AWS Glue TableInput for an external Delta table."""
    if not table_name:
        raise ValueError("AWS Glue table name is required.")

    if not isinstance(schema, StructType):
        raise TypeError("Spark schema must be a StructType.")

    normalized_location = _normalize_s3_location(location)

    table_input: dict[str, Any] = {
        "Name": table_name,
        "TableType": "EXTERNAL_TABLE",
        "StorageDescriptor": {
            "Columns": spark_schema_to_glue_columns(schema),
            "Location": normalized_location,
        },
        "Parameters": {
            "classification": "delta",
            "delta.table": "true",
        },
    }

    if description is not None:
        table_input["Description"] = description

    return table_input
