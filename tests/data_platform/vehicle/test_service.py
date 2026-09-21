from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from data_platform.vehicle.service import VehicleDataService


class _FakeSparkColumn:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __ge__(self, other):
        return ("ge", self.name, other)

    def __le__(self, other):
        return ("le", self.name, other)

    def asc(self):
        return ("asc", self.name)


def _build_service(*, rows=None):
    reader = MagicMock()
    spark = MagicMock()
    dataframe = MagicMock()

    reader.read.return_value = dataframe

    dataframe.vehicle_id = _FakeSparkColumn("vehicle_id")
    dataframe.event_time = _FakeSparkColumn("event_time")

    dataframe.select.return_value = dataframe
    dataframe.where.return_value = dataframe
    dataframe.orderBy.return_value = dataframe
    dataframe.limit.return_value = dataframe
    dataframe.collect.return_value = rows or []

    service = VehicleDataService(
        reader=reader,
        spark=spark,
    )

    return service, reader, spark, dataframe


def test_query_invokes_reader_and_selects_supported_columns():
    service, reader, spark, dataframe = _build_service()

    result = service.query(
        vehicle_id="veh-123",
        limit=50,
    )

    reader.read.assert_called_once_with(spark)
    dataframe.select.assert_called_once_with(
        "vehicle_id",
        "event_time",
        "speed",
    )
    dataframe.limit.assert_called_once_with(50)

    assert result["source"] == "silver.vehicle_events"
    assert result["count"] == 0
    assert result["records"] == []


def test_query_applies_vehicle_filter():
    service, _, _, dataframe = _build_service()

    service.query(
        vehicle_id="  veh-123  ",
        limit=10,
    )

    assert dataframe.where.call_count == 1

    vehicle_filter = dataframe.where.call_args.args[0]

    assert vehicle_filter is not None


def test_query_applies_start_and_end_filters():
    service, _, _, dataframe = _build_service()

    start = datetime(
        2026,
        1,
        1,
        0,
        0,
        tzinfo=timezone.utc,
    )
    end = datetime(
        2026,
        1,
        2,
        0,
        0,
        tzinfo=timezone.utc,
    )

    service.query(
        start_time=start,
        end_time=end,
        limit=20,
    )

    assert dataframe.where.call_count == 2


def test_query_applies_vehicle_and_time_filters():
    service, _, _, dataframe = _build_service()

    start = datetime(
        2026,
        1,
        1,
        0,
        0,
        tzinfo=timezone.utc,
    )
    end = datetime(
        2026,
        1,
        2,
        0,
        0,
        tzinfo=timezone.utc,
    )

    service.query(
        vehicle_id="veh-123",
        start_time=start,
        end_time=end,
        limit=25,
    )

    assert dataframe.where.call_count == 3


def test_query_applies_deterministic_ordering_and_limit():
    service, _, _, dataframe = _build_service()

    service.query(limit=37)

    dataframe.orderBy.assert_called_once()
    dataframe.limit.assert_called_once_with(37)

    order_arguments = dataframe.orderBy.call_args.args

    assert len(order_arguments) == 2


def test_query_returns_bounded_response_structure():
    event_time = datetime(
        2026,
        1,
        1,
        12,
        30,
        tzinfo=timezone.utc,
    )

    rows = [
        {
            "vehicle_id": "veh-123",
            "event_time": event_time,
            "speed": 52.5,
        }
    ]

    service, _, _, _ = _build_service(rows=rows)

    result = service.query(
        vehicle_id="veh-123",
        start_time=event_time,
        end_time=event_time,
        limit=50,
    )

    assert set(result) == {
        "source",
        "filters",
        "count",
        "records",
    }

    assert result["source"] == "silver.vehicle_events"
    assert result["count"] == 1
    assert result["records"] == [
        {
            "vehicle_id": "veh-123",
            "event_time": event_time.isoformat(),
            "speed": 52.5,
        }
    ]

    assert result["filters"] == {
        "vehicle_id": "veh-123",
        "start_time": event_time.isoformat(),
        "end_time": event_time.isoformat(),
        "limit": 50,
    }


def test_query_returns_empty_result_when_no_rows_match():
    service, _, _, dataframe = _build_service(rows=[])

    result = service.query(
        vehicle_id="veh-does-not-exist",
        limit=10,
    )

    assert dataframe.collect.called
    assert result["count"] == 0
    assert result["records"] == []


def test_query_rejects_empty_vehicle_id():
    service, _, _, _ = _build_service()

    with pytest.raises(
        ValueError,
        match="vehicle_id must not be empty",
    ):
        service.query(vehicle_id="   ")


def test_query_rejects_non_string_vehicle_id():
    service, _, _, _ = _build_service()

    with pytest.raises(
        TypeError,
        match="vehicle_id must be a string",
    ):
        service.query(vehicle_id=123)


def test_query_rejects_invalid_start_time_type():
    service, _, _, _ = _build_service()

    with pytest.raises(
        TypeError,
        match="start_time must be a datetime",
    ):
        service.query(start_time="2026-01-01")


def test_query_rejects_invalid_end_time_type():
    service, _, _, _ = _build_service()

    with pytest.raises(
        TypeError,
        match="end_time must be a datetime",
    ):
        service.query(end_time="2026-01-01")


def test_query_rejects_reversed_date_range():
    service, _, _, _ = _build_service()

    start = datetime(
        2026,
        1,
        2,
        tzinfo=timezone.utc,
    )
    end = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    with pytest.raises(
        ValueError,
        match="start_time must be before or equal to end_time",
    ):
        service.query(
            start_time=start,
            end_time=end,
        )


@pytest.mark.parametrize("limit", [0, -1, 101, 1000])
def test_query_rejects_limit_outside_allowed_range(limit):
    service, _, _, _ = _build_service()

    with pytest.raises(
        ValueError,
        match="limit must be between 1 and 100",
    ):
        service.query(limit=limit)


@pytest.mark.parametrize("limit", [True, False, 50.0, "50", None])
def test_query_rejects_non_integer_limit(limit):
    service, _, _, _ = _build_service()

    with pytest.raises(
        TypeError,
        match="limit must be an integer",
    ):
        service.query(limit=limit)


def test_query_accepts_minimum_and_maximum_limits():
    service, _, _, dataframe = _build_service()

    service.query(limit=1)
    service.query(limit=100)

    assert dataframe.limit.call_count == 2
    assert dataframe.limit.call_args_list[0].args == (1,)
    assert dataframe.limit.call_args_list[1].args == (100,)


def test_query_preserves_null_event_time_in_record():
    rows = [
        {
            "vehicle_id": "veh-123",
            "event_time": None,
            "speed": 10.0,
        }
    ]

    service, _, _, _ = _build_service(rows=rows)

    result = service.query(
        vehicle_id="veh-123",
        limit=1,
    )

    assert result["records"] == [
        {
            "vehicle_id": "veh-123",
            "event_time": None,
            "speed": 10.0,
        }
    ]
