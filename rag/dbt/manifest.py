from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rag.dbt.models import DbtColumn, DbtModel


class DbtManifestParser:
    """Parses dbt manifest metadata into canonical DbtModel objects."""

    def parse(
        self,
        manifest: Mapping[str, Any],
    ) -> Sequence[DbtModel]:
        models: list[DbtModel] = []

        for unique_id, node in manifest.get("nodes", {}).items():
            if node.get("resource_type") != "model":
                continue

            models.append(
                self._parse_model(
                    unique_id=unique_id,
                    node=node,
                )
            )

        return models

    @staticmethod
    def _parse_model(
        *,
        unique_id: str,
        node: Mapping[str, Any],
    ) -> DbtModel:
        config = node.get("config") or {}

        columns = tuple(
            DbtColumn(
                name=name,
                description=column.get("description") or "",
                tags=tuple(column.get("tags") or ()),
                meta=dict(column.get("meta") or {}),
            )
            for name, column in (node.get("columns") or {}).items()
        )

        depends_on = tuple(node.get("depends_on", {}).get("nodes") or ())

        return DbtModel(
            unique_id=unique_id,
            name=node.get("name") or "",
            resource_type=node.get("resource_type") or "model",
            database=node.get("database"),
            schema=node.get("schema"),
            alias=node.get("alias"),
            description=node.get("description") or "",
            materialization=config.get("materialized"),
            tags=tuple(node.get("tags") or ()),
            meta=dict(node.get("meta") or {}),
            depends_on=depends_on,
            columns=columns,
        )
