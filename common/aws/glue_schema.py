from __future__ import annotations

from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructType,
    TimestampType,
)

_SPARK_TO_GLUE_TYPE = {
    StringType: "string",
    TimestampType: "timestamp",
    DoubleType: "double",
    IntegerType: "int",
    LongType: "bigint",
}


def spark_schema_to_glue_columns(schema: StructType) -> list[dict[str, str]]:
    """Convert a Spark StructType into AWS Glue column definitions."""
    columns: list[dict[str, str]] = []

    for field in schema.fields:
        glue_type = None

        for spark_type, candidate in _SPARK_TO_GLUE_TYPE.items():
            if isinstance(field.dataType, spark_type):
                glue_type = candidate
                break

        if glue_type is None:
            raise ValueError(
                f"Unsupported Spark data type for column "
                f"'{field.name}': {field.dataType.simpleString()}"
            )

        columns.append(
            {
                "Name": field.name,
                "Type": glue_type,
            }
        )

    return columns
