from __future__ import annotations

from datetime import datetime

from rag.loaders import VehicleMetricsDocumentMapper


def test_vehicle_metrics_mapper_creates_canonical_document(spark):
    dataframe = spark.createDataFrame(
        [
            (
                "V001",
                3,
                48.166666666666664,
                45.5,
                51.8,
                datetime(2026, 1, 1, 10, 0, 0),
                datetime(2026, 1, 1, 12, 0, 0),
            ),
        ],
        [
            "vehicle_id",
            "event_count",
            "avg_speed",
            "min_speed",
            "max_speed",
            "first_event_time",
            "last_event_time",
        ],
    )

    documents = VehicleMetricsDocumentMapper.loader().load(dataframe)

    assert len(documents) == 1

    document = documents[0]

    assert document.id == "vehicle:V001"

    assert (
        document.content == "Vehicle V001 recorded 3 events. "
        "Average speed was 48.17. "
        "Minimum speed was 45.50. "
        "Maximum speed was 51.80. "
        "The observation period started at "
        "2026-01-01T10:00:00 "
        "and ended at "
        "2026-01-01T12:00:00."
    )

    assert document.metadata == {
        "source": "gold.vehicle_metrics",
        "data_layer": "gold",
        "dataset": "vehicle_metrics",
        "entity_type": "vehicle",
        "vehicle_id": "V001",
    }


def test_vehicle_metrics_mapper_creates_one_document_per_vehicle(spark):
    dataframe = spark.createDataFrame(
        [
            (
                "V001",
                3,
                48.17,
                45.5,
                51.8,
                datetime(2026, 1, 1, 10, 0, 0),
                datetime(2026, 1, 1, 12, 0, 0),
            ),
            (
                "V002",
                3,
                34.73,
                32.1,
                36.4,
                datetime(2026, 1, 2, 10, 0, 0),
                datetime(2026, 1, 2, 12, 0, 0),
            ),
        ],
        [
            "vehicle_id",
            "event_count",
            "avg_speed",
            "min_speed",
            "max_speed",
            "first_event_time",
            "last_event_time",
        ],
    )

    documents = VehicleMetricsDocumentMapper.loader().load(dataframe)

    assert [document.id for document in documents] == [
        "vehicle:V001",
        "vehicle:V002",
    ]

    assert [document.metadata["vehicle_id"] for document in documents] == [
        "V001",
        "V002",
    ]


def test_vehicle_metrics_mapper_metadata_is_retrieval_friendly(spark):
    dataframe = spark.createDataFrame(
        [
            (
                "V001",
                3,
                48.17,
                45.5,
                51.8,
                datetime(2026, 1, 1, 10, 0, 0),
                datetime(2026, 1, 1, 12, 0, 0),
            ),
        ],
        [
            "vehicle_id",
            "event_count",
            "avg_speed",
            "min_speed",
            "max_speed",
            "first_event_time",
            "last_event_time",
        ],
    )

    document = VehicleMetricsDocumentMapper.loader().load(dataframe)[0]

    assert document.metadata["source"] == "gold.vehicle_metrics"
    assert document.metadata["data_layer"] == "gold"
    assert document.metadata["dataset"] == "vehicle_metrics"
    assert document.metadata["entity_type"] == "vehicle"
    assert document.metadata["vehicle_id"] == "V001"
