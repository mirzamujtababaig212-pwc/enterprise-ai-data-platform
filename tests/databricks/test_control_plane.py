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
