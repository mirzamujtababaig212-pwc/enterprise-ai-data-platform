from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from common.snowflake.control_plane import SnowflakeControlPlaneClient


def test_constructor_uses_injected_connection():
    connection = Mock()

    control_plane = SnowflakeControlPlaneClient(connection=connection)

    assert control_plane._connection is connection


def test_constructor_requires_snowflake_account(monkeypatch):
    monkeypatch.setattr(
        "common.snowflake.control_plane.Settings.snowflake.options",
        lambda: {
            "sfUser": "test-user",
            "sfPassword": "test-password",
        },
    )

    with pytest.raises(ValueError, match="SNOWFLAKE_ACCOUNT is required."):
        SnowflakeControlPlaneClient()


def test_constructor_requires_snowflake_user(monkeypatch):
    monkeypatch.setattr(
        "common.snowflake.control_plane.Settings.snowflake.options",
        lambda: {
            "sfURL": "account.example",
            "sfPassword": "test-password",
        },
    )

    with pytest.raises(ValueError, match="SNOWFLAKE_USER is required."):
        SnowflakeControlPlaneClient()


def test_constructor_requires_snowflake_password(monkeypatch):
    monkeypatch.setattr(
        "common.snowflake.control_plane.Settings.snowflake.options",
        lambda: {
            "sfURL": "account.example",
            "sfUser": "test-user",
        },
    )

    with pytest.raises(ValueError, match="SNOWFLAKE_PASSWORD is required."):
        SnowflakeControlPlaneClient()


def test_constructor_creates_connector_connection(monkeypatch):
    connection = Mock()
    connect = Mock(return_value=connection)

    monkeypatch.setattr(
        "common.snowflake.control_plane.Settings.snowflake.options",
        lambda: {
            "sfURL": "account.example",
            "sfUser": "test-user",
            "sfPassword": "test-password",
            "sfDatabase": "VEHICLE_PLATFORM",
            "sfSchema": "ANALYTICS",
            "sfWarehouse": "COMPUTE_WH",
            "sfRole": "DATA_PLATFORM",
        },
    )
    monkeypatch.setattr(
        "common.snowflake.control_plane.snowflake.connector.connect",
        connect,
    )

    control_plane = SnowflakeControlPlaneClient()

    assert control_plane._connection is connection
    connect.assert_called_once_with(
        account="account.example",
        user="test-user",
        password="test-password",
        database="VEHICLE_PLATFORM",
        schema="ANALYTICS",
        warehouse="COMPUTE_WH",
        role="DATA_PLATFORM",
    )


def test_get_table_metadata_maps_describe_columns():
    connection = Mock()
    cursor = Mock()

    number_type = SimpleNamespace(name="NUMBER")
    float_type = SimpleNamespace(name="FLOAT")

    cursor.describe.return_value = [
        SimpleNamespace(
            name="vehicle_id",
            type_code=number_type,
            precision=38,
            scale=0,
            is_nullable=False,
        ),
        SimpleNamespace(
            name="avg_speed",
            type_code=float_type,
            precision=None,
            scale=None,
            is_nullable=True,
        ),
    ]

    connection.cursor.return_value = cursor

    control_plane = SnowflakeControlPlaneClient(connection=connection)

    metadata = control_plane.get_table_metadata("VEHICLE_PLATFORM.ANALYTICS.RPT_VEHICLE_SUMMARY")

    assert metadata.full_name == ("VEHICLE_PLATFORM.ANALYTICS.RPT_VEHICLE_SUMMARY")
    assert metadata.database == "VEHICLE_PLATFORM"
    assert metadata.schema == "ANALYTICS"
    assert metadata.name == "RPT_VEHICLE_SUMMARY"

    assert len(metadata.columns) == 2

    assert metadata.columns[0].name == "vehicle_id"
    assert metadata.columns[0].type_name == "NUMBER"
    assert metadata.columns[0].type_text == "NUMBER(38,0)"
    assert metadata.columns[0].nullable is False
    assert metadata.columns[0].position == 1

    assert metadata.columns[1].name == "avg_speed"
    assert metadata.columns[1].type_name == "FLOAT"
    assert metadata.columns[1].type_text == "FLOAT"
    assert metadata.columns[1].nullable is True
    assert metadata.columns[1].position == 2

    cursor.describe.assert_called_once_with(
        "SELECT * FROM VEHICLE_PLATFORM.ANALYTICS.RPT_VEHICLE_SUMMARY"
    )
    cursor.close.assert_called_once_with()


def test_get_table_metadata_supports_two_part_name():
    connection = Mock()
    cursor = Mock()

    cursor.describe.return_value = []

    connection.cursor.return_value = cursor

    control_plane = SnowflakeControlPlaneClient(connection=connection)

    metadata = control_plane.get_table_metadata("ANALYTICS.VEHICLE_EVENTS")

    assert metadata.database is None
    assert metadata.schema == "ANALYTICS"
    assert metadata.name == "VEHICLE_EVENTS"
    assert metadata.columns == ()


def test_get_table_metadata_supports_one_part_name():
    connection = Mock()
    cursor = Mock()

    cursor.describe.return_value = []

    connection.cursor.return_value = cursor

    control_plane = SnowflakeControlPlaneClient(connection=connection)

    metadata = control_plane.get_table_metadata("VEHICLE_EVENTS")

    assert metadata.database is None
    assert metadata.schema is None
    assert metadata.name == "VEHICLE_EVENTS"


def test_invalid_table_name_is_rejected():
    connection = Mock()
    cursor = Mock()
    cursor.describe.return_value = []
    connection.cursor.return_value = cursor

    control_plane = SnowflakeControlPlaneClient(connection=connection)

    with pytest.raises(
        ValueError,
        match="one to three dot-separated parts",
    ):
        control_plane.get_table_metadata("A.B.C.D")
