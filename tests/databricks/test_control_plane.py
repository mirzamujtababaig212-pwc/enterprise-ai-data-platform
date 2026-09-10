from types import SimpleNamespace
from unittest.mock import Mock

from common.databricks.control_plane import DatabricksControlPlaneClient


def test_list_catalogs_delegates_to_workspace_client():
    client = Mock()
    client.catalogs.list.return_value = ["catalog-a", "catalog-b"]

    control_plane = DatabricksControlPlaneClient(client=client)

    result = control_plane.list_catalogs()

    assert result == ["catalog-a", "catalog-b"]
    client.catalogs.list.assert_called_once_with()


def test_get_catalog_delegates_to_workspace_client():
    client = Mock()
    catalog = Mock()
    client.catalogs.get.return_value = catalog

    control_plane = DatabricksControlPlaneClient(client=client)

    result = control_plane.get_catalog("main")

    assert result is catalog
    client.catalogs.get.assert_called_once_with("main")


def test_list_schemas_delegates_to_workspace_client():
    client = Mock()
    client.schemas.list.return_value = ["analytics", "reporting"]

    control_plane = DatabricksControlPlaneClient(client=client)

    result = control_plane.list_schemas("main")

    assert result == ["analytics", "reporting"]
    client.schemas.list.assert_called_once_with("main")


def test_get_schema_delegates_to_workspace_client():
    client = Mock()
    schema = Mock()
    client.schemas.get.return_value = schema

    control_plane = DatabricksControlPlaneClient(client=client)

    result = control_plane.get_schema("main.analytics")

    assert result is schema
    client.schemas.get.assert_called_once_with("main.analytics")


def test_list_tables_delegates_to_workspace_client():
    client = Mock()
    client.tables.list.return_value = ["vehicle_events", "vehicle_summary"]

    control_plane = DatabricksControlPlaneClient(client=client)

    result = control_plane.list_tables("main", "analytics")

    assert result == ["vehicle_events", "vehicle_summary"]
    client.tables.list.assert_called_once_with("main", "analytics")


def test_get_table_delegates_to_workspace_client():
    client = Mock()
    table = Mock()
    client.tables.get.return_value = table

    control_plane = DatabricksControlPlaneClient(client=client)

    result = control_plane.get_table("main.analytics.vehicle_events")

    assert result is table
    client.tables.get.assert_called_once_with("main.analytics.vehicle_events")


def test_constructor_requires_databricks_host(monkeypatch):
    monkeypatch.setattr(
        "common.databricks.control_plane.Settings.databricks.options",
        lambda: {"token": "test-token"},
    )

    try:
        DatabricksControlPlaneClient()
    except ValueError as exc:
        assert str(exc) == "DATABRICKS_HOST is required."
    else:
        raise AssertionError("Expected ValueError for missing DATABRICKS_HOST")


def test_constructor_requires_databricks_token(monkeypatch):
    monkeypatch.setattr(
        "common.databricks.control_plane.Settings.databricks.options",
        lambda: {"host": "https://workspace.example.com"},
    )

    try:
        DatabricksControlPlaneClient()
    except ValueError as exc:
        assert str(exc) == "DATABRICKS_TOKEN is required."
    else:
        raise AssertionError("Expected ValueError for missing DATABRICKS_TOKEN")


def test_constructor_creates_workspace_client(monkeypatch):
    workspace_client = Mock()

    monkeypatch.setattr(
        "common.databricks.control_plane.Settings.databricks.options",
        lambda: {
            "host": "https://workspace.example.com",
            "token": "test-token",
        },
    )
    monkeypatch.setattr(
        "common.databricks.control_plane.WorkspaceClient",
        Mock(return_value=workspace_client),
    )

    control_plane = DatabricksControlPlaneClient()

    assert control_plane._client is workspace_client


def test_get_table_metadata_maps_table_and_columns():
    client = Mock()

    column_a = SimpleNamespace(
        name="vehicle_id",
        type_name=SimpleNamespace(value="STRING"),
        type_text="string",
        nullable=False,
        comment="Vehicle identifier",
        position=0,
        partition_index=None,
    )
    column_b = SimpleNamespace(
        name="avg_speed",
        type_name=SimpleNamespace(value="DOUBLE"),
        type_text="double",
        nullable=True,
        comment="Average speed",
        position=1,
        partition_index=None,
    )

    table = SimpleNamespace(
        full_name="vehicle_platform.analytics.rpt_vehicle_summary",
        catalog_name="vehicle_platform",
        schema_name="analytics",
        name="rpt_vehicle_summary",
        table_type=SimpleNamespace(value="MANAGED"),
        owner="data-platform",
        comment="Vehicle fleet summary",
        table_id="table-123",
        properties={"domain": "vehicle", "tier": "gold"},
        columns=[column_a, column_b],
    )

    client.tables.get.return_value = table

    control_plane = DatabricksControlPlaneClient(client=client)

    metadata = control_plane.get_table_metadata("vehicle_platform.analytics.rpt_vehicle_summary")

    assert metadata.full_name == "vehicle_platform.analytics.rpt_vehicle_summary"
    assert metadata.catalog == "vehicle_platform"
    assert metadata.schema == "analytics"
    assert metadata.name == "rpt_vehicle_summary"
    assert metadata.table_type == "MANAGED"
    assert metadata.owner == "data-platform"
    assert metadata.comment == "Vehicle fleet summary"
    assert metadata.table_id == "table-123"
    assert metadata.properties == {
        "domain": "vehicle",
        "tier": "gold",
    }

    assert len(metadata.columns) == 2
    assert metadata.columns[0].name == "vehicle_id"
    assert metadata.columns[0].type_name == "STRING"
    assert metadata.columns[0].nullable is False
    assert metadata.columns[1].name == "avg_speed"
    assert metadata.columns[1].type_name == "DOUBLE"

    client.tables.get.assert_called_once_with("vehicle_platform.analytics.rpt_vehicle_summary")


def test_get_table_metadata_handles_missing_optional_fields():
    client = Mock()

    table = SimpleNamespace(
        full_name="main.analytics.vehicle_events",
        catalog_name="main",
        schema_name="analytics",
        name="vehicle_events",
        table_type=None,
        owner=None,
        comment=None,
        table_id=None,
        properties=None,
        columns=None,
    )

    client.tables.get.return_value = table

    control_plane = DatabricksControlPlaneClient(client=client)

    metadata = control_plane.get_table_metadata("main.analytics.vehicle_events")

    assert metadata.full_name == "main.analytics.vehicle_events"
    assert metadata.table_type is None
    assert metadata.owner is None
    assert metadata.properties == {}
    assert metadata.columns == ()
