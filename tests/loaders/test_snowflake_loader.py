from unittest.mock import Mock

from rag.loaders.snowflake import SnowflakeDocumentLoader
from rag.models import Document
from common.snowflake.metadata import SnowflakeTableMetadata


def test_snowflake_document_loader_enriches_documents_with_source_provenance():
    table_metadata = SnowflakeTableMetadata(
        full_name="VEHICLE_PLATFORM.PUBLIC.RPT_VEHICLE_SUMMARY",
        database="VEHICLE_PLATFORM",
        schema="PUBLIC",
        name="RPT_VEHICLE_SUMMARY",
        table_type="TABLE",
    )

    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_count": 25,
                        "avg_speed": 48.17,
                    }
                )
            )
        ]
    )

    reader = Mock()
    reader.read.return_value = dataframe

    loader = SnowflakeDocumentLoader(
        metadata=table_metadata,
        reader=reader,
        id_fn=lambda row: f"fleet:{row['vehicle_count']}",
        content_fn=lambda row: (
            f"Snowflake fleet summary: {row['vehicle_count']} vehicles "
            f"with average speed {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "snowflake",
            "dataset": table_metadata.name,
            "vehicle_count": row["vehicle_count"],
            "avg_speed": row["avg_speed"],
        },
    )

    spark = Mock()

    documents = loader.load(spark)

    assert documents == [
        Document(
            id="fleet:25",
            content="Snowflake fleet summary: 25 vehicles with average speed 48.17.",
            metadata={
                "source": "snowflake",
                "dataset": "RPT_VEHICLE_SUMMARY",
                "vehicle_count": 25,
                "avg_speed": 48.17,
                "source_ref": {
                    "platform": "snowflake",
                    "object_type": "table",
                    "object_name": "RPT_VEHICLE_SUMMARY",
                    "namespace": "VEHICLE_PLATFORM.PUBLIC",
                },
            },
        )
    ]

    reader.read.assert_called_once_with(spark)
