from common.aws.catalog_publisher import AwsGlueCatalogPublisher


class FakeGlueClient:
    def __init__(self):
        self.create_calls = []
        self.update_calls = []

    def create_table(self, **kwargs):
        self.create_calls.append(kwargs)
        return {"Table": kwargs["TableInput"]}

    def update_table(self, **kwargs):
        self.update_calls.append(kwargs)
        return {"Table": kwargs["TableInput"]}


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


def test_create_table_publishes_table_input():
    client = FakeGlueClient()
    publisher = AwsGlueCatalogPublisher(client=client)

    response = publisher.create_table(
        database_name="enterprise_ai_platform",
        table_input=_table_input(),
    )

    assert response["Table"]["Name"] == "vehicle_events"

    assert client.create_calls == [
        {
            "DatabaseName": "enterprise_ai_platform",
            "TableInput": _table_input(),
        }
    ]


def test_create_table_includes_catalog_id_when_provided():
    client = FakeGlueClient()
    publisher = AwsGlueCatalogPublisher(client=client)

    publisher.create_table(
        database_name="enterprise_ai_platform",
        table_input=_table_input(),
        catalog_id="715621342004",
    )

    assert client.create_calls == [
        {
            "CatalogId": "715621342004",
            "DatabaseName": "enterprise_ai_platform",
            "TableInput": _table_input(),
        }
    ]


def test_update_table_publishes_table_input():
    client = FakeGlueClient()
    publisher = AwsGlueCatalogPublisher(client=client)

    publisher.update_table(
        database_name="enterprise_ai_platform",
        table_input=_table_input(),
    )

    assert client.update_calls == [
        {
            "DatabaseName": "enterprise_ai_platform",
            "TableInput": _table_input(),
        }
    ]


def test_create_table_requires_database_name():
    publisher = AwsGlueCatalogPublisher(client=FakeGlueClient())

    try:
        publisher.create_table(
            database_name="",
            table_input=_table_input(),
        )
    except ValueError as exc:
        assert str(exc) == "AWS Glue database name is required."
    else:
        raise AssertionError("Expected ValueError")


def test_create_table_requires_table_name():
    publisher = AwsGlueCatalogPublisher(client=FakeGlueClient())

    table_input = _table_input()
    table_input["Name"] = ""

    try:
        publisher.create_table(
            database_name="enterprise_ai_platform",
            table_input=table_input,
        )
    except ValueError as exc:
        assert str(exc) == "AWS Glue table name is required."
    else:
        raise AssertionError("Expected ValueError")


def test_create_table_accepts_built_delta_table_input():
    from pyspark.sql.types import StringType, StructField, StructType

    from common.aws.glue_table_input import build_delta_table_input

    client = FakeGlueClient()
    publisher = AwsGlueCatalogPublisher(client=client)

    schema = StructType(
        [
            StructField("vehicle_id", StringType(), True),
        ]
    )

    table_input = build_delta_table_input(
        table_name="vehicle_events",
        schema=schema,
        location="s3a://enterprise-data-ai-platform/bronze/vehicle_events",
    )

    publisher.create_table(
        database_name="enterprise_ai_platform",
        table_input=table_input,
    )

    assert len(client.create_calls) == 1

    request = client.create_calls[0]

    assert request["DatabaseName"] == "enterprise_ai_platform"
    assert request["TableInput"] == table_input
