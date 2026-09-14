from __future__ import annotations

from typing import Any

import boto3

from common.aws.metadata import (
    AwsGlueColumnMetadata,
    AwsGlueTableMetadata,
)


class AwsGlueControlPlaneClient:
    """
    Read-only AWS Glue Data Catalog control-plane client.

    The control plane is responsible for AWS-native metadata discovery.
    Actual table data access remains the responsibility of the appropriate
    data-plane reader or Spark integration.

    The boto3 client is injectable so unit tests do not require AWS
    credentials or network access.
    """

    def __init__(
        self,
        client: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        self._client = client or boto3.client("glue", region_name=region_name)

    def get_table(
        self,
        database_name: str,
        table_name: str,
        catalog_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get the raw AWS Glue table response."""

        if not database_name:
            raise ValueError("AWS Glue database name is required.")

        if not table_name:
            raise ValueError("AWS Glue table name is required.")

        request: dict[str, Any] = {
            "DatabaseName": database_name,
            "Name": table_name,
        }

        if catalog_id:
            request["CatalogId"] = catalog_id

        request.update(kwargs)

        return self._client.get_table(**request)

    def get_table_metadata(
        self,
        database_name: str,
        table_name: str,
        catalog_id: str | None = None,
        **kwargs: Any,
    ) -> AwsGlueTableMetadata:
        """
        Get canonical DELDAI metadata for an AWS Glue table.

        AWS SDK response objects are converted at the control-plane
        boundary so downstream components do not depend directly on
        boto3 response structures.
        """

        response = self.get_table(
            database_name=database_name,
            table_name=table_name,
            catalog_id=catalog_id,
            **kwargs,
        )

        table = response.get("Table") or {}

        storage_descriptor = table.get("StorageDescriptor") or {}

        columns = tuple(
            AwsGlueColumnMetadata(
                name=column.get("Name"),
                type_name=column.get("Type"),
                comment=column.get("Comment"),
                position=index,
            )
            for index, column in enumerate(
                storage_descriptor.get("Columns") or [],
                start=1,
            )
        )

        partition_columns = tuple(
            AwsGlueColumnMetadata(
                name=column.get("Name"),
                type_name=column.get("Type"),
                comment=column.get("Comment"),
                partition_index=index,
            )
            for index, column in enumerate(
                table.get("PartitionKeys") or [],
                start=1,
            )
        )

        return AwsGlueTableMetadata(
            catalog_id=table.get("CatalogId") or catalog_id,
            database_name=table.get("DatabaseName") or database_name,
            name=table.get("Name") or table_name,
            table_type=table.get("TableType"),
            owner=table.get("Owner"),
            description=table.get("Description"),
            location=storage_descriptor.get("Location"),
            input_format=storage_descriptor.get("InputFormat"),
            output_format=storage_descriptor.get("OutputFormat"),
            serde_library=(storage_descriptor.get("SerdeInfo") or {}).get("SerializationLibrary"),
            parameters={
                str(key): str(value) for key, value in (table.get("Parameters") or {}).items()
            },
            columns=columns + partition_columns,
        )
