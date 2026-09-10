from common.fabric.control_plane import FabricControlPlaneClient
from common.fabric.metadata import FabricTableMetadata


def test_constructor_requires_workspace():
    try:
        FabricControlPlaneClient(
            workspace="",
            lakehouse="lakehouse",
            access_token="token",
        )
    except ValueError as exc:
        assert str(exc) == "Fabric workspace is required."
    else:
        raise AssertionError("Expected ValueError")


def test_constructor_requires_lakehouse():
    try:
        FabricControlPlaneClient(
            workspace="workspace",
            lakehouse="",
            access_token="token",
        )
    except ValueError as exc:
        assert str(exc) == "Fabric lakehouse is required."
    else:
        raise AssertionError("Expected ValueError")


def test_live_request_requires_access_token():
    client = FabricControlPlaneClient(
        workspace="workspace",
        lakehouse="lakehouse",
        access_token=None,
    )

    try:
        client.list_schemas()
    except ValueError as exc:
        assert str(exc) == ("FABRIC_ACCESS_TOKEN is required for live Fabric requests.")
    else:
        raise AssertionError("Expected ValueError")


def test_list_schemas_maps_response():
    calls = []

    def fake_get(path, params, access_token):
        calls.append(
            {
                "path": path,
                "params": params,
                "access_token": access_token,
            }
        )

        return {
            "schemas": [
                {
                    "name": "dbo",
                    "catalog_name": "SalesLakehouse.Lakehouse",
                    "full_name": "SalesLakehouse.Lakehouse.dbo",
                }
            ],
            "next_page_token": None,
        }

    client = FabricControlPlaneClient(
        workspace="workspace-id",
        lakehouse="SalesLakehouse",
        access_token="test-token",
        request_get=fake_get,
    )

    schemas = client.list_schemas()

    assert schemas[0]["name"] == "dbo"

    assert calls == [
        {
            "path": ("/workspace-id/SalesLakehouse.Lakehouse" "/api/2.1/unity-catalog/schemas"),
            "params": {
                "catalog_name": "SalesLakehouse.Lakehouse",
            },
            "access_token": "test-token",
        }
    ]


def test_list_tables_maps_response():
    calls = []

    def fake_get(path, params, access_token):
        calls.append(
            {
                "path": path,
                "params": params,
                "access_token": access_token,
            }
        )

        return {
            "tables": [
                {
                    "name": "vehicle_events",
                    "catalog_name": "VehicleLakehouse.Lakehouse",
                    "schema_name": "dbo",
                    "table_type": "MANAGED",
                    "data_source_format": "DELTA",
                    "storage_location": ("https://onelake.example/Tables/vehicle_events"),
                }
            ],
            "next_page_token": None,
        }

    client = FabricControlPlaneClient(
        workspace="workspace-id",
        lakehouse="VehicleLakehouse",
        access_token="test-token",
        request_get=fake_get,
    )

    tables = client.list_tables()

    assert tables[0]["name"] == "vehicle_events"
    assert tables[0]["data_source_format"] == "DELTA"

    assert calls == [
        {
            "path": ("/workspace-id/VehicleLakehouse.Lakehouse" "/api/2.1/unity-catalog/tables"),
            "params": {
                "catalog_name": "VehicleLakehouse.Lakehouse",
                "schema_name": "dbo",
            },
            "access_token": "test-token",
        }
    ]


def test_list_tables_follows_pagination():
    calls = []

    def fake_get(path, params, access_token):
        calls.append(params)

        if len(calls) == 1:
            return {
                "tables": [
                    {
                        "name": "vehicle_events",
                    }
                ],
                "next_page_token": "page-2",
            }

        return {
            "tables": [
                {
                    "name": "vehicle_summary",
                }
            ],
            "next_page_token": None,
        }

    client = FabricControlPlaneClient(
        workspace="workspace-id",
        lakehouse="VehicleLakehouse",
        access_token="test-token",
        request_get=fake_get,
    )

    tables = client.list_tables()

    assert [table["name"] for table in tables] == [
        "vehicle_events",
        "vehicle_summary",
    ]

    assert calls == [
        {
            "catalog_name": "VehicleLakehouse.Lakehouse",
            "schema_name": "dbo",
        },
        {
            "catalog_name": "VehicleLakehouse.Lakehouse",
            "schema_name": "dbo",
            "page_token": "page-2",
        },
    ]


def test_get_table_metadata_maps_columns():
    def fake_get(path, params, access_token):
        assert path == (
            "/workspace-id/VehicleLakehouse.Lakehouse"
            "/api/2.1/unity-catalog/tables/"
            "VehicleLakehouse.Lakehouse.dbo.vehicle_events"
        )
        assert params is None
        assert access_token == "test-token"

        return {
            "name": "vehicle_events",
            "catalog_name": "VehicleLakehouse.Lakehouse",
            "schema_name": "dbo",
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "storage_location": ("https://onelake.example/Tables/vehicle_events"),
            "comment": "Vehicle telemetry",
            "owner": "platform",
            "table_id": "table-123",
            "properties": {
                "domain": "mobility",
            },
            "columns": [
                {
                    "name": "vehicle_id",
                    "type_name": "string",
                    "type_text": None,
                    "nullable": False,
                    "comment": "Vehicle identifier",
                    "position": 0,
                    "partition_index": 0,
                },
                {
                    "name": "speed",
                    "type_name": "double",
                    "type_text": None,
                    "nullable": True,
                    "comment": None,
                    "position": 1,
                    "partition_index": 0,
                },
            ],
        }

    client = FabricControlPlaneClient(
        workspace="workspace-id",
        lakehouse="VehicleLakehouse",
        access_token="test-token",
        request_get=fake_get,
    )

    metadata = client.get_table_metadata(
        table_name="vehicle_events",
    )

    assert isinstance(metadata, FabricTableMetadata)

    assert metadata.workspace_id is None
    assert metadata.workspace_name == "workspace-id"
    assert metadata.lakehouse_id is None
    assert metadata.lakehouse_name == "VehicleLakehouse"
    assert metadata.catalog_name == "VehicleLakehouse.Lakehouse"
    assert metadata.schema_name == "dbo"
    assert metadata.table_id == "table-123"
    assert metadata.table_name == "vehicle_events"
    assert metadata.table_type == "MANAGED"
    assert metadata.format == "DELTA"
    assert metadata.comment == "Vehicle telemetry"
    assert metadata.owner == "platform"

    assert metadata.properties == {
        "domain": "mobility",
    }

    assert len(metadata.columns) == 2

    assert metadata.columns[0].name == "vehicle_id"
    assert metadata.columns[0].type_name == "string"
    assert metadata.columns[0].nullable is False
    assert metadata.columns[0].position == 0

    assert metadata.columns[1].name == "speed"
    assert metadata.columns[1].type_name == "double"
    assert metadata.columns[1].nullable is True
    assert metadata.columns[1].position == 1


def test_catalog_name_preserves_lakehouse_suffix():
    client = FabricControlPlaneClient(
        workspace="workspace",
        lakehouse="Sales.Lakehouse",
        access_token="token",
    )

    assert client._catalog_name() == "Sales.Lakehouse"


def test_catalog_name_preserves_uuid():
    lakehouse_id = "98765432-dcba-4209-8ac2-0821c7f8bd91"

    client = FabricControlPlaneClient(
        workspace="workspace",
        lakehouse=lakehouse_id,
        access_token="token",
    )

    assert client._catalog_name() == lakehouse_id


def test_table_metadata_preserves_friendly_names_without_fake_ids():
    def fake_get(path, params, access_token):
        return {
            "name": "vehicle_events",
            "catalog_name": "VehicleLakehouse.Lakehouse",
            "schema_name": "dbo",
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "columns": [],
        }

    client = FabricControlPlaneClient(
        workspace="VehicleWorkspace",
        lakehouse="VehicleLakehouse",
        access_token="test-token",
        request_get=fake_get,
    )

    metadata = client.get_table_metadata("vehicle_events")

    assert metadata.workspace_id is None
    assert metadata.workspace_name == "VehicleWorkspace"

    assert metadata.lakehouse_id is None
    assert metadata.lakehouse_name == "VehicleLakehouse"


def test_table_metadata_preserves_guid_identifiers_without_fake_names():
    workspace_id = "12345678-abcd-4fbd-9e50-3937d8eb1915"
    lakehouse_id = "98765432-dcba-4209-8ac2-0821c7f8bd91"

    def fake_get(path, params, access_token):
        assert path == (
            f"/{workspace_id}/{lakehouse_id}"
            "/api/2.1/unity-catalog/tables/"
            f"{lakehouse_id}.dbo.vehicle_events"
        )

        return {
            "name": "vehicle_events",
            "catalog_name": lakehouse_id,
            "schema_name": "dbo",
            "table_type": "MANAGED",
            "data_source_format": "DELTA",
            "columns": [],
        }

    client = FabricControlPlaneClient(
        workspace=workspace_id,
        lakehouse=lakehouse_id,
        access_token="test-token",
        request_get=fake_get,
    )

    metadata = client.get_table_metadata("vehicle_events")

    assert metadata.workspace_id == workspace_id
    assert metadata.workspace_name is None

    assert metadata.lakehouse_id == lakehouse_id
    assert metadata.lakehouse_name is None
