from botocore.exceptions import ClientError

from common.aws.catalog_publisher import AwsGlueCatalogPublisher
from common.aws.catalog_synchronizer import AwsGlueCatalogSynchronizer
from common.aws.control_plane import AwsGlueControlPlaneClient


def _table_input():
    return {
        "Name": "vehicle_events",
        "TableType": "EXTERNAL_TABLE",
        "Description": "Vehicle event Bronze Delta table.",
        "StorageDescriptor": {
            "Columns": [
                {"Name": "vehicle_id", "Type": "string"},
                {"Name": "event_time", "Type": "timestamp"},
            ],
            "Location": "s3://enterprise-data-ai-platform/bronze/vehicle_events",
        },
        "Parameters": {
            "classification": "delta",
            "delta.table": "true",
        },
    }


class FakeGlueClient:
    def __init__(self, table_response=None):
        self.table_response = table_response
        self.get_calls = []
        self.create_calls = []
        self.update_calls = []

    def get_table(self, **kwargs):
        self.get_calls.append(kwargs)

        if isinstance(self.table_response, Exception):
            raise self.table_response

        return self.table_response

    def create_table(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"Table": kwargs["TableInput"]}

    def update_table(self, **kwargs):
        self.update_calls.append(kwargs)
        return {"Table": kwargs["TableInput"]}


def _not_found_error():
    return ClientError(
        {
            "Error": {
                "Code": "EntityNotFoundException",
                "Message": "Table not found",
            }
        },
        "GetTable",
    )


def _existing_table(table_input):
    return {
        "Table": {
            "CatalogId": "715621342004",
            "DatabaseName": "enterprise_ai_platform",
            "Name": table_input["Name"],
            "TableType": table_input["TableType"],
            "Description": table_input["Description"],
            "StorageDescriptor": {
                "Columns": table_input["StorageDescriptor"]["Columns"],
                "Location": table_input["StorageDescriptor"]["Location"],
            },
            "Parameters": table_input["Parameters"],
        }
    }


def _synchronizer(client):
    control_plane = AwsGlueControlPlaneClient(client=client)
    publisher = AwsGlueCatalogPublisher(client=client)

    return AwsGlueCatalogSynchronizer(
        control_plane=control_plane,
        publisher=publisher,
    )


def test_sync_creates_missing_table():
    client = FakeGlueClient(table_response=_not_found_error())

    synchronizer = _synchronizer(client)

    result = synchronizer.sync(
        database_name="enterprise_ai_platform",
        table_input=_table_input(),
    )

    assert result == "created"
    assert len(client.get_calls) == 1
    assert len(client.create_calls) == 1
    assert len(client.update_calls) == 0

    assert client.create_calls[0] == {
        "DatabaseName": "enterprise_ai_platform",
        "TableInput": _table_input(),
    }


def test_sync_is_noop_when_table_matches():
    table_input = _table_input()
    client = FakeGlueClient(table_response=_existing_table(table_input))

    synchronizer = _synchronizer(client)

    result = synchronizer.sync(
        database_name="enterprise_ai_platform",
        table_input=table_input,
    )

    assert result == "unchanged"
    assert len(client.get_calls) == 1
    assert len(client.create_calls) == 0
    assert len(client.update_calls) == 0


def test_sync_updates_when_schema_drifts():
    desired = _table_input()

    actual = _existing_table(desired)
    actual["Table"]["StorageDescriptor"]["Columns"] = [
        {"Name": "vehicle_id", "Type": "string"},
    ]

    client = FakeGlueClient(table_response=actual)

    synchronizer = _synchronizer(client)

    result = synchronizer.sync(
        database_name="enterprise_ai_platform",
        table_input=desired,
    )

    assert result == "updated"
    assert len(client.get_calls) == 1
    assert len(client.create_calls) == 0
    assert len(client.update_calls) == 1

    assert client.update_calls[0] == {
        "DatabaseName": "enterprise_ai_platform",
        "TableInput": desired,
    }


def test_sync_updates_when_location_drifts():
    desired = _table_input()

    actual = _existing_table(desired)
    actual["Table"]["StorageDescriptor"][
        "Location"
    ] = "s3://enterprise-data-ai-platform/bronze/old_location"

    client = FakeGlueClient(table_response=actual)

    synchronizer = _synchronizer(client)

    result = synchronizer.sync(
        database_name="enterprise_ai_platform",
        table_input=desired,
    )

    assert result == "updated"
    assert len(client.create_calls) == 0
    assert len(client.update_calls) == 1


def test_sync_requires_database_name():
    client = FakeGlueClient(table_response=_not_found_error())
    synchronizer = _synchronizer(client)

    try:
        synchronizer.sync(
            database_name="",
            table_input=_table_input(),
        )
    except ValueError as exc:
        assert str(exc) == "AWS Glue database name is required."
    else:
        raise AssertionError("Expected ValueError")


def test_sync_requires_table_name():
    client = FakeGlueClient(table_response=_not_found_error())
    synchronizer = _synchronizer(client)

    table_input = _table_input()
    table_input["Name"] = ""

    try:
        synchronizer.sync(
            database_name="enterprise_ai_platform",
            table_input=table_input,
        )
    except ValueError as exc:
        assert str(exc) == "AWS Glue table name is required."
    else:
        raise AssertionError("Expected ValueError")


def test_sync_preserves_catalog_id():
    table_input = _table_input()
    client = FakeGlueClient(table_response=_existing_table(table_input))

    synchronizer = _synchronizer(client)

    result = synchronizer.sync(
        database_name="enterprise_ai_platform",
        table_input=table_input,
        catalog_id="715621342004",
    )

    assert result == "unchanged"

    assert client.get_calls == [
        {
            "DatabaseName": "enterprise_ai_platform",
            "Name": "vehicle_events",
            "CatalogId": "715621342004",
        }
    ]


def test_sync_propagates_unexpected_glue_error():
    error = ClientError(
        {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "Access denied",
            }
        },
        "GetTable",
    )

    client = FakeGlueClient(table_response=error)
    synchronizer = _synchronizer(client)

    try:
        synchronizer.sync(
            database_name="enterprise_ai_platform",
            table_input=_table_input(),
        )
    except ClientError as exc:
        assert exc.response["Error"]["Code"] == "AccessDeniedException"
    else:
        raise AssertionError("Expected AccessDeniedException")

    assert len(client.create_calls) == 0
    assert len(client.update_calls) == 0
