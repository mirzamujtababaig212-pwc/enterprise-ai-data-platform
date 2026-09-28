from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from tools.execution.context import ToolExecutionContext
from tools.vehicle.data_query import VehicleDataQueryTool


def _build_tool():
    service = MagicMock()
    service.query.return_value = {
        "source": "silver.vehicle_events",
        "filters": {
            "vehicle_id": "veh-123",
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-02T00:00:00+00:00",
            "limit": 50,
        },
        "count": 1,
        "records": [
            {
                "vehicle_id": "veh-123",
                "event_time": "2026-01-01T12:00:00+00:00",
                "speed": 52.5,
            }
        ],
    }

    return VehicleDataQueryTool(service), service


def test_definition_name_is_vehicle_data_query():
    tool, _ = _build_tool()

    assert tool.definition.name == "vehicle.data.query"
    assert tool.definition.provider is not None
    assert tool.definition.provider.kind == "native"
    assert tool.definition.provider.name == "internal"


def test_definition_has_expected_input_schema():
    tool, _ = _build_tool()

    schema = tool.definition.input_schema

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    assert set(schema["properties"]) == {
        "vehicle_id",
        "start_time",
        "end_time",
        "limit",
    }

    assert schema["properties"]["vehicle_id"]["type"] == "string"

    assert schema["properties"]["start_time"] == {
        "type": "string",
        "format": "date-time",
        "description": "Optional inclusive event start time.",
    }

    assert schema["properties"]["end_time"] == {
        "type": "string",
        "format": "date-time",
        "description": "Optional inclusive event end time.",
    }

    assert schema["properties"]["limit"]["type"] == "integer"
    assert schema["properties"]["limit"]["minimum"] == 1
    assert schema["properties"]["limit"]["maximum"] == 100
    assert schema["properties"]["limit"]["default"] == 50

    assert "required" not in schema


def test_definition_metadata_is_enterprise_data_read_only():
    tool, _ = _build_tool()

    metadata = tool.definition.metadata

    assert metadata["category"] == "enterprise_data"
    assert metadata["read_only"] is True
    assert "authorization_required" not in metadata


def test_definition_has_no_tool_retries():
    tool, _ = _build_tool()

    assert tool.definition.execution_policy.max_retries == 0


@pytest.mark.asyncio
async def test_execute_uses_default_limit_of_50():
    tool, service = _build_tool()

    await tool.execute(
        {
            "vehicle_id": "veh-123",
        }
    )

    service.query.assert_called_once_with(
        vehicle_id="veh-123",
        start_time=None,
        end_time=None,
        limit=50,
    )


@pytest.mark.asyncio
async def test_execute_parses_iso8601_datetimes():
    tool, service = _build_tool()

    await tool.execute(
        {
            "vehicle_id": "veh-123",
            "start_time": "2026-01-01T00:00:00+00:00",
            "end_time": "2026-01-02T00:00:00+00:00",
            "limit": 25,
        }
    )

    service.query.assert_called_once()

    call = service.query.call_args.kwargs

    assert call["vehicle_id"] == "veh-123"
    assert call["start_time"] == datetime(
        2026,
        1,
        1,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert call["end_time"] == datetime(
        2026,
        1,
        2,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert call["limit"] == 25


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("start_time", "not-a-date"),
        ("end_time", "not-a-date"),
    ],
)
async def test_execute_rejects_invalid_iso8601_datetime(
    field_name,
    value,
):
    tool, service = _build_tool()

    arguments = {
        "vehicle_id": "veh-123",
        field_name: value,
    }

    with pytest.raises(
        ValueError,
        match="ISO-8601 datetime string",
    ):
        await tool.execute(arguments)

    service.query.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "value", "expected_exception"),
    [
        ("start_time", "", TypeError),
        ("start_time", "   ", TypeError),
        ("start_time", 123, TypeError),
        ("end_time", "", TypeError),
        ("end_time", "   ", TypeError),
        ("end_time", 123, TypeError),
    ],
)
async def test_execute_rejects_invalid_datetime_values(
    field_name,
    value,
    expected_exception,
):
    tool, service = _build_tool()

    with pytest.raises(
        expected_exception,
        match="ISO-8601 datetime string",
    ):
        await tool.execute(
            {
                field_name: value,
            }
        )

    service.query.assert_not_called()


@pytest.mark.asyncio
async def test_execute_delegates_service_result():
    tool, service = _build_tool()

    expected = service.query.return_value

    result = await tool.execute(
        {
            "vehicle_id": "veh-123",
            "limit": 10,
        }
    )

    assert result is expected
    service.query.assert_called_once_with(
        vehicle_id="veh-123",
        start_time=None,
        end_time=None,
        limit=10,
    )


@pytest.mark.asyncio
async def test_execute_with_context_delegates_to_same_execution_path():
    tool, service = _build_tool()

    context = ToolExecutionContext(
        run_id="run-123",
        call_id="call-123",
        agent_name="enterprise-rag-analyst",
        session_id="session-123",
        user_id="enterprise-demo-user",
    )

    result = await tool.execute_with_context(
        {
            "vehicle_id": "veh-123",
            "limit": 5,
        },
        context,
    )

    assert result is service.query.return_value

    service.query.assert_called_once_with(
        vehicle_id="veh-123",
        start_time=None,
        end_time=None,
        limit=5,
    )


@pytest.mark.asyncio
async def test_execute_accepts_optional_vehicle_and_time_filters():
    tool, service = _build_tool()

    await tool.execute(
        {
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-01T23:59:59Z",
        }
    )

    call = service.query.call_args.kwargs

    assert call["vehicle_id"] is None
    assert call["start_time"] == datetime(
        2026,
        1,
        1,
        0,
        0,
        tzinfo=timezone.utc,
    )
    assert call["end_time"] == datetime(
        2026,
        1,
        1,
        23,
        59,
        59,
        tzinfo=timezone.utc,
    )
    assert call["limit"] == 50


@pytest.mark.asyncio
async def test_execute_propagates_service_validation_errors():
    tool, service = _build_tool()

    service.query.side_effect = ValueError("limit must be between 1 and 100.")

    with pytest.raises(
        ValueError,
        match="limit must be between 1 and 100",
    ):
        await tool.execute(
            {
                "vehicle_id": "veh-123",
                "limit": 101,
            }
        )


def test_tool_does_not_require_an_authorization_metadata_flag():
    tool, _ = _build_tool()

    assert "authorization_required" not in tool.definition.metadata


def test_tool_context_type_is_supported():
    tool, _ = _build_tool()

    context = ToolExecutionContext(
        run_id="run-123",
        call_id="call-123",
        user_id="enterprise-demo-user",
    )

    assert context.user_id == "enterprise-demo-user"
