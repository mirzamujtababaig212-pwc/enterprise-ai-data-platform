from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

from common.fabric.metadata import (
    FabricColumnMetadata,
    FabricTableMetadata,
)


class FabricControlPlaneClient:
    """
    Read-only Microsoft Fabric control-plane client.

    The client uses the OneLake Delta Table API for metadata discovery.
    Authentication is intentionally kept outside the metadata model and
    can be supplied through an environment variable.

    HTTP requests are injectable so tests do not require live Fabric
    credentials or network access.
    """

    BASE_URL = "https://onelake.table.fabric.microsoft.com/delta"

    def __init__(
        self,
        workspace: str,
        lakehouse: str,
        access_token: str | None = None,
        request_get: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        if not workspace:
            raise ValueError("Fabric workspace is required.")

        if not lakehouse:
            raise ValueError("Fabric lakehouse is required.")

        self.workspace = workspace
        self.lakehouse = lakehouse
        self.access_token = access_token or os.getenv("FABRIC_ACCESS_TOKEN")
        self._request_get = request_get or self._default_request_get

    def list_schemas(self) -> list[dict[str, Any]]:
        """
        List schemas available in the configured Fabric lakehouse.
        """

        path = (
            f"/{self._quote(self.workspace)}"
            f"/{self._quote(self._data_item_name())}"
            "/api/2.1/unity-catalog/schemas"
        )

        return self._get_paginated(
            path,
            params={
                "catalog_name": self._catalog_name(),
            },
            collection_key="schemas",
        )

    def list_tables(
        self,
        schema_name: str = "dbo",
    ) -> list[dict[str, Any]]:
        """
        List tables in a Fabric lakehouse schema.
        """

        if not schema_name:
            raise ValueError("Fabric schema name is required.")

        path = (
            f"/{self._quote(self.workspace)}"
            f"/{self._quote(self._data_item_name())}"
            "/api/2.1/unity-catalog/tables"
        )

        return self._get_paginated(
            path,
            params={
                "catalog_name": self._catalog_name(),
                "schema_name": schema_name,
            },
            collection_key="tables",
        )

    def get_table_metadata(
        self,
        table_name: str,
        schema_name: str = "dbo",
    ) -> FabricTableMetadata:
        """
        Get canonical DELDAI metadata for a Fabric table.
        """

        if not table_name:
            raise ValueError("Fabric table name is required.")

        if not schema_name:
            raise ValueError("Fabric schema name is required.")

        full_table_name = f"{self._catalog_name()}.{schema_name}.{table_name}"

        path = (
            f"/{self._quote(self.workspace)}"
            f"/{self._quote(self._data_item_name())}"
            "/api/2.1/unity-catalog/tables/"
            f"{self._quote(full_table_name, safe='.')}"
        )

        response = self._get(path)

        columns = tuple(
            FabricColumnMetadata(
                name=column.get("name"),
                type_name=column.get("type_name"),
                type_text=column.get("type_text"),
                nullable=column.get("nullable"),
                comment=column.get("comment"),
                position=column.get("position"),
                partition_index=column.get("partition_index"),
            )
            for column in response.get("columns") or []
        )

        properties = response.get("properties") or {}

        return FabricTableMetadata(
            workspace_id=self._workspace_id(),
            workspace_name=self._workspace_name(),
            lakehouse_id=self._lakehouse_id(),
            lakehouse_name=self._lakehouse_name(),
            catalog_name=response.get("catalog_name"),
            schema_name=response.get("schema_name"),
            table_id=response.get("table_id"),
            table_name=response.get("name") or table_name,
            table_type=response.get("table_type"),
            location=response.get("storage_location"),
            format=response.get("data_source_format"),
            owner=response.get("owner"),
            comment=response.get("comment"),
            properties={str(key): str(value) for key, value in properties.items()},
            columns=columns,
        )

    def _get_paginated(
        self,
        path: str,
        params: dict[str, str] | None,
        collection_key: str,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        next_page_token: str | None = None

        while True:
            request_params = dict(params or {})

            if next_page_token:
                request_params["page_token"] = next_page_token

            response = self._get(
                path,
                params=request_params,
            )

            items.extend(response.get(collection_key) or [])

            next_page_token = response.get("next_page_token")

            if not next_page_token:
                break

        return items

    def _get(
        self,
        path: str,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self.access_token:
            raise ValueError("FABRIC_ACCESS_TOKEN is required for live Fabric requests.")

        return self._request_get(
            path=path,
            params=params,
            access_token=self.access_token,
        )

    def _workspace_id(self) -> str | None:
        if self._looks_like_uuid(self.workspace):
            return self.workspace

        return None

    def _workspace_name(self) -> str | None:
        if self._looks_like_uuid(self.workspace):
            return None

        return self.workspace

    def _lakehouse_id(self) -> str | None:
        if self._looks_like_uuid(self.lakehouse):
            return self.lakehouse

        return None

    def _lakehouse_name(self) -> str | None:
        if self._looks_like_uuid(self.lakehouse):
            return None

        if self.lakehouse.lower().endswith(".lakehouse"):
            return self.lakehouse[: -len(".lakehouse")]

        return self.lakehouse

    def _catalog_name(self) -> str:
        """
        Return the OneLake catalog identifier.

        Friendly Lakehouse names use the Fabric item type suffix.
        GUID identifiers are passed through unchanged.
        """

        if self.lakehouse.lower().endswith(".lakehouse"):
            return self.lakehouse

        if self._looks_like_uuid(self.lakehouse):
            return self.lakehouse

        return f"{self.lakehouse}.Lakehouse"

    def _data_item_name(self) -> str:
        """
        Return the data-item path component expected by OneLake.
        """

        return self._catalog_name()

    @staticmethod
    def _looks_like_uuid(value: str) -> bool:
        parts = value.split("-")

        return (
            len(parts) == 5
            and len(parts[0]) == 8
            and len(parts[1]) == 4
            and len(parts[2]) == 4
            and len(parts[3]) == 4
            and len(parts[4]) == 12
        )

    @staticmethod
    def _quote(value: str, safe: str = "") -> str:
        return quote(value, safe=safe)

    @classmethod
    def _default_request_get(
        cls,
        path: str,
        params: dict[str, str] | None,
        access_token: str,
    ) -> dict[str, Any]:
        url = f"{cls.BASE_URL}{path}"

        if params:
            query = "&".join(
                f"{quote(str(key))}={quote(str(value))}" for key, value in params.items()
            )
            url = f"{url}?{query}"

        request = Request(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            method="GET",
        )

        with urlopen(request) as response:
            return json.load(response)
