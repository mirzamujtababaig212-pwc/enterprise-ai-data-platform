from __future__ import annotations

from typing import Any

import snowflake.connector

from common.config.settings import Settings
from common.snowflake.metadata import (
    SnowflakeColumnMetadata,
    SnowflakeTableMetadata,
)


class SnowflakeControlPlaneClient:
    """
    Read-only Snowflake control-plane client.

    The control plane is responsible for Snowflake metadata.
    Actual table data access remains the responsibility of
    SnowflakeReader and the Spark data plane.
    """

    def __init__(self, connection: Any | None = None) -> None:
        if connection is not None:
            self._connection = connection
            return

        options = Settings.snowflake.options()

        account = options.get("sfURL")
        user = options.get("sfUser")
        password = options.get("sfPassword")

        if not account:
            raise ValueError("SNOWFLAKE_ACCOUNT is required.")
        if not user:
            raise ValueError("SNOWFLAKE_USER is required.")
        if not password:
            raise ValueError("SNOWFLAKE_PASSWORD is required.")

        connection_options = {
            "account": account,
            "user": user,
            "password": password,
        }

        database = options.get("sfDatabase")
        schema = options.get("sfSchema")
        warehouse = options.get("sfWarehouse")
        role = options.get("sfRole")

        if database:
            connection_options["database"] = database
        if schema:
            connection_options["schema"] = schema
        if warehouse:
            connection_options["warehouse"] = warehouse
        if role:
            connection_options["role"] = role

        self._connection = snowflake.connector.connect(**connection_options)

    def get_table_metadata(self, full_name: str) -> SnowflakeTableMetadata:
        """
        Get canonical DELDAI metadata for a Snowflake table.

        Metadata is converted at the control-plane boundary so downstream
        components do not depend directly on Snowflake connector objects.
        """

        database, schema, name = self._split_table_name(full_name)

        cursor = self._connection.cursor()

        try:
            column_metadata = cursor.describe(f"SELECT * FROM {full_name}")

            columns = tuple(
                SnowflakeColumnMetadata(
                    name=getattr(column, "name", None),
                    type_name=self._type_name(column),
                    type_text=self._type_text(column),
                    nullable=getattr(column, "is_nullable", None),
                    position=index,
                )
                for index, column in enumerate(column_metadata, start=1)
            )
        finally:
            cursor.close()

        return SnowflakeTableMetadata(
            full_name=full_name,
            database=database,
            schema=schema,
            name=name,
            columns=columns,
        )

    @staticmethod
    def _split_table_name(full_name: str) -> tuple[str | None, str | None, str | None]:
        parts = [part.strip() for part in full_name.split(".")]

        if len(parts) == 3:
            return parts[0], parts[1], parts[2]

        if len(parts) == 2:
            return None, parts[0], parts[1]

        if len(parts) == 1:
            return None, None, parts[0]

        raise ValueError("Snowflake table name must contain one to three dot-separated parts.")

    @staticmethod
    def _type_name(column: Any) -> str | None:
        type_code = getattr(column, "type_code", None)

        if type_code is None:
            return None

        return getattr(type_code, "name", None) or str(type_code)

    @staticmethod
    def _type_text(column: Any) -> str | None:
        type_name = SnowflakeControlPlaneClient._type_name(column)

        precision = getattr(column, "precision", None)
        scale = getattr(column, "scale", None)

        if type_name is None:
            return None

        if precision is not None and scale is not None:
            return f"{type_name}({precision},{scale})"

        if precision is not None:
            return f"{type_name}({precision})"

        return type_name
