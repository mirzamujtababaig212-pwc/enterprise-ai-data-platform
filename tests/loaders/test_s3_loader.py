from __future__ import annotations

from unittest.mock import Mock

from rag.loaders import S3DocumentLoader
from rag.models import Document


def test_s3_document_loader_enriches_documents_with_source_provenance():
    dataframe = Mock()
    dataframe.toLocalIterator.return_value = iter(
        [
            Mock(
                asDict=Mock(
                    return_value={
                        "vehicle_id": "V001",
                        "avg_speed": 48.17,
                    }
                )
            )
        ]
    )

    reader = Mock()
    reader.read.return_value = dataframe

    loader = S3DocumentLoader(
        reader=reader,
        path="s3://test-bucket/bronze/vehicle_events",
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (f"Vehicle {row['vehicle_id']} average speed {row['avg_speed']}"),
        metadata_fn=lambda row: {
            "source": "s3.vehicle_events",
            "data_layer": "bronze",
            "dataset": "vehicle_events",
            "vehicle_id": row["vehicle_id"],
        },
    )

    spark = Mock()

    documents = loader.load(spark)

    assert documents == [
        Document(
            id="vehicle:V001",
            content="Vehicle V001 average speed 48.17",
            metadata={
                "source": "s3.vehicle_events",
                "data_layer": "bronze",
                "dataset": "vehicle_events",
                "vehicle_id": "V001",
                "source_ref": {
                    "platform": "aws",
                    "object_type": "s3_path",
                    "object_name": "s3://test-bucket/bronze/vehicle_events",
                    "namespace": "s3://test-bucket",
                },
            },
        )
    ]

    reader.read.assert_called_once_with(spark)
