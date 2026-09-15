from __future__ import annotations

from typing import Any

import boto3


class AwsGlueCatalogPublisher:
    """
    AWS Glue Data Catalog publisher.

    This component owns catalog mutations only. Metadata discovery remains
    the responsibility of AwsGlueControlPlaneClient.
    """

    def __init__(
        self,
        client: Any | None = None,
        region_name: str | None = None,
    ) -> None:
        self._client = client or boto3.client("glue", region_name=region_name)

    @staticmethod
    def _validate_table_input(
        database_name: str,
        table_input: dict[str, Any],
    ) -> None:
        if not database_name:
            raise ValueError("AWS Glue database name is required.")

        if not table_input.get("Name"):
            raise ValueError("AWS Glue table name is required.")

    def create_table(
        self,
        database_name: str,
        table_input: dict[str, Any],
        catalog_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an AWS Glue Data Catalog table."""

        self._validate_table_input(database_name, table_input)

        request: dict[str, Any] = {
            "DatabaseName": database_name,
            "TableInput": table_input,
        }

        if catalog_id:
            request["CatalogId"] = catalog_id

        return self._client.create_table(**request)

    def update_table(
        self,
        database_name: str,
        table_input: dict[str, Any],
        catalog_id: str | None = None,
    ) -> dict[str, Any]:
        """Update an AWS Glue Data Catalog table."""

        self._validate_table_input(database_name, table_input)

        request: dict[str, Any] = {
            "DatabaseName": database_name,
            "TableInput": table_input,
        }

        if catalog_id:
            request["CatalogId"] = catalog_id

        return self._client.update_table(**request)
