from __future__ import annotations

from typing import Any

from databricks.sdk import WorkspaceClient

from common.config.settings import Settings


class DatabricksControlPlaneClient:
    """
    Read-only Databricks control-plane client.

    The control plane is responsible for workspace and Unity Catalog
    metadata. Actual table data access remains the responsibility of
    DatabricksReader and the Spark data plane.
    """

    def __init__(self, client: Any | None = None) -> None:
        if client is not None:
            self._client = client
            return

        options = Settings.databricks.options()
        host = options.get("host")
        token = options.get("token")

        if not host:
            raise ValueError("DATABRICKS_HOST is required.")
        if not token:
            raise ValueError("DATABRICKS_TOKEN is required.")

        self._client = WorkspaceClient(
            host=host,
            token=token,
        )

    def list_catalogs(self, **kwargs):
        """List catalogs visible to the authenticated workspace identity."""
        return self._client.catalogs.list(**kwargs)

    def get_catalog(self, name: str, **kwargs):
        """Get a catalog by name."""
        return self._client.catalogs.get(name, **kwargs)

    def list_schemas(self, catalog_name: str, **kwargs):
        """List schemas within a catalog."""
        return self._client.schemas.list(catalog_name, **kwargs)

    def get_schema(self, full_name: str, **kwargs):
        """Get a schema using its three-level or fully qualified name."""
        return self._client.schemas.get(full_name, **kwargs)

    def list_tables(self, catalog_name: str, schema_name: str, **kwargs):
        """List tables within a catalog and schema."""
        return self._client.tables.list(catalog_name, schema_name, **kwargs)

    def get_table(self, full_name: str, **kwargs):
        """Get table metadata using its fully qualified name."""
        return self._client.tables.get(full_name, **kwargs)
