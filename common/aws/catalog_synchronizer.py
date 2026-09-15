from __future__ import annotations

from typing import Any

from botocore.exceptions import ClientError

from common.aws.catalog_publisher import AwsGlueCatalogPublisher
from common.aws.control_plane import AwsGlueControlPlaneClient


class AwsGlueCatalogSynchronizer:
    """
    Synchronize a desired AWS Glue table definition with the catalog.

    Metadata discovery is delegated to the read-only control plane client.
    Catalog mutations are delegated to the mutation-only publisher.
    """

    def __init__(
        self,
        control_plane: AwsGlueControlPlaneClient,
        publisher: AwsGlueCatalogPublisher,
    ) -> None:
        self._control_plane = control_plane
        self._publisher = publisher

    @staticmethod
    def _validate_table_input(
        database_name: str,
        table_input: dict[str, Any],
    ) -> None:
        if not database_name:
            raise ValueError("AWS Glue database name is required.")

        if not table_input.get("Name"):
            raise ValueError("AWS Glue table name is required.")

    @staticmethod
    def _normalize_location(location: str | None) -> str | None:
        if location is None:
            return None

        normalized = str(location).rstrip("/")

        if normalized.startswith("s3a://"):
            return f"s3://{normalized[6:]}"

        return normalized

    @classmethod
    def _table_matches(
        cls,
        metadata: Any,
        table_input: dict[str, Any],
    ) -> bool:
        storage_descriptor = table_input.get("StorageDescriptor") or {}

        desired_columns = tuple(
            (column.get("Name"), column.get("Type"))
            for column in storage_descriptor.get("Columns") or []
        )

        actual_columns = tuple(
            (column.name, column.type_name)
            for column in metadata.columns
            if column.partition_index is None
        )

        desired_location = cls._normalize_location(storage_descriptor.get("Location"))
        actual_location = cls._normalize_location(metadata.location)

        desired_parameters = {
            str(key): str(value) for key, value in (table_input.get("Parameters") or {}).items()
        }

        actual_parameters = {
            str(key): str(value) for key, value in (metadata.parameters or {}).items()
        }

        return (
            metadata.name == table_input.get("Name")
            and metadata.table_type == table_input.get("TableType")
            and metadata.description == table_input.get("Description")
            and actual_location == desired_location
            and actual_columns == desired_columns
            and actual_parameters == desired_parameters
        )

    @staticmethod
    def _is_not_found_error(exc: ClientError) -> bool:
        error = exc.response.get("Error") or {}
        return error.get("Code") == "EntityNotFoundException"

    def sync(
        self,
        database_name: str,
        table_input: dict[str, Any],
        catalog_id: str | None = None,
    ) -> str:
        """
        Synchronize a Glue table.

        Returns:
            "created" when a missing table is created.
            "unchanged" when the catalog already matches.
            "updated" when catalog drift is reconciled.
        """
        self._validate_table_input(database_name, table_input)

        table_name = table_input["Name"]

        try:
            metadata = self._control_plane.get_table_metadata(
                database_name=database_name,
                table_name=table_name,
                catalog_id=catalog_id,
            )
        except ClientError as exc:
            if not self._is_not_found_error(exc):
                raise

            self._publisher.create_table(
                database_name=database_name,
                table_input=table_input,
                catalog_id=catalog_id,
            )

            return "created"

        if self._table_matches(metadata, table_input):
            return "unchanged"

        self._publisher.update_table(
            database_name=database_name,
            table_input=table_input,
            catalog_id=catalog_id,
        )

        return "updated"
