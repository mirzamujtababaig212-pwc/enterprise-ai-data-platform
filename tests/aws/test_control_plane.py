from common.aws.control_plane import AwsGlueControlPlaneClient


class FakeGlueClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get_table(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_get_table_metadata_maps_glue_response():
    client = FakeGlueClient(
        {
            "Table": {
                "CatalogId": "715621342004",
                "DatabaseName": "enterprise_ai",
                "Name": "vehicle_events",
                "TableType": "EXTERNAL_TABLE",
                "Owner": "data-platform",
                "Description": "Vehicle event data",
                "StorageDescriptor": {
                    "Location": ("s3://enterprise-data-ai-platform/" "bronze/vehicle_events"),
                    "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                    "OutputFormat": ("org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"),
                    "SerdeInfo": {
                        "SerializationLibrary": (
                            "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe"
                        )
                    },
                    "Columns": [
                        {
                            "Name": "vehicle_id",
                            "Type": "string",
                            "Comment": "Vehicle identifier",
                        },
                        {
                            "Name": "event_ts",
                            "Type": "timestamp",
                        },
                    ],
                },
                "PartitionKeys": [
                    {
                        "Name": "event_date",
                        "Type": "date",
                    }
                ],
                "Parameters": {
                    "classification": "parquet",
                    "delta.table": "true",
                },
            }
        }
    )

    glue = AwsGlueControlPlaneClient(client=client)

    metadata = glue.get_table_metadata(
        database_name="enterprise_ai",
        table_name="vehicle_events",
        catalog_id="715621342004",
    )

    assert metadata.catalog_id == "715621342004"
    assert metadata.database_name == "enterprise_ai"
    assert metadata.name == "vehicle_events"
    assert metadata.table_type == "EXTERNAL_TABLE"
    assert metadata.owner == "data-platform"
    assert metadata.description == "Vehicle event data"
    assert metadata.location == "s3://enterprise-data-ai-platform/bronze/vehicle_events"
    assert metadata.input_format.endswith("TextInputFormat")
    assert metadata.output_format.endswith("HiveIgnoreKeyTextOutputFormat")
    assert metadata.serde_library.endswith("LazySimpleSerDe")

    assert metadata.parameters == {
        "classification": "parquet",
        "delta.table": "true",
    }

    assert len(metadata.columns) == 3

    assert metadata.columns[0].name == "vehicle_id"
    assert metadata.columns[0].type_name == "string"
    assert metadata.columns[0].position == 1
    assert metadata.columns[0].partition_index is None

    assert metadata.columns[2].name == "event_date"
    assert metadata.columns[2].type_name == "date"
    assert metadata.columns[2].position is None
    assert metadata.columns[2].partition_index == 1

    assert client.calls == [
        {
            "DatabaseName": "enterprise_ai",
            "Name": "vehicle_events",
            "CatalogId": "715621342004",
        }
    ]


def test_get_table_metadata_uses_requested_identity_when_response_omits_it():
    client = FakeGlueClient(
        {
            "Table": {
                "StorageDescriptor": {},
            }
        }
    )

    glue = AwsGlueControlPlaneClient(client=client)

    metadata = glue.get_table_metadata(
        database_name="bronze",
        table_name="vehicle_events",
    )

    assert metadata.database_name == "bronze"
    assert metadata.name == "vehicle_events"


def test_get_table_requires_database_name():
    glue = AwsGlueControlPlaneClient(client=FakeGlueClient({}))

    try:
        glue.get_table("", "vehicle_events")
    except ValueError as exc:
        assert str(exc) == "AWS Glue database name is required."
    else:
        raise AssertionError("Expected ValueError")


def test_get_table_requires_table_name():
    glue = AwsGlueControlPlaneClient(client=FakeGlueClient({}))

    try:
        glue.get_table("bronze", "")
    except ValueError as exc:
        assert str(exc) == "AWS Glue table name is required."
    else:
        raise AssertionError("Expected ValueError")
