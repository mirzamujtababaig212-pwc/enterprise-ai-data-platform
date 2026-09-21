from __future__ import annotations

from datetime import datetime
from typing import Any

from data_platform.vehicle.service import VehicleDataService
from tools.execution.context import ToolExecutionContext
from tools.models import ToolDefinition, ToolExecutionPolicy


class VehicleDataQueryTool:
    """Bounded read-only vehicle telemetry query tool."""

    def __init__(self, service: VehicleDataService) -> None:
        self._service = service

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vehicle.data.query",
            description=(
                "Query bounded vehicle telemetry evidence from the canonical "
                "enterprise vehicle-events dataset."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "vehicle_id": {
                        "type": "string",
                        "description": "Optional vehicle identifier.",
                    },
                    "start_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional inclusive event start time.",
                    },
                    "end_time": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Optional inclusive event end time.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50,
                        "description": "Maximum number of telemetry events.",
                    },
                },
                "additionalProperties": False,
            },
            metadata={
                "category": "enterprise_data",
                "read_only": True,
            },
            execution_policy=ToolExecutionPolicy(
                max_retries=0,
            ),
        )

    async def execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return self._execute(arguments)

    async def execute_with_context(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return self._execute(arguments)

    def _execute(
        self,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        vehicle_id = arguments.get("vehicle_id")

        start_time = self._parse_datetime(
            arguments.get("start_time"),
            field_name="start_time",
        )

        end_time = self._parse_datetime(
            arguments.get("end_time"),
            field_name="end_time",
        )

        limit = arguments.get("limit", 50)

        return self._service.query(
            vehicle_id=vehicle_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

    @staticmethod
    def _parse_datetime(
        value: Any,
        *,
        field_name: str,
    ) -> datetime | None:
        if value is None:
            return None

        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"vehicle.data.query {field_name} must be an ISO-8601 datetime string.")

        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"vehicle.data.query {field_name} must be an ISO-8601 datetime string."
            ) from exc
