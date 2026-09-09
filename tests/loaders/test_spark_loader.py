from __future__ import annotations

import pytest

from rag.loaders import SparkDataFrameDocumentLoader


def test_spark_dataframe_loader_creates_documents(spark):
    dataframe = spark.createDataFrame(
        [
            ("V001", 3, 48.5),
            ("V002", 2, 34.7),
        ],
        ["vehicle_id", "event_count", "avg_speed"],
    )

    loader = SparkDataFrameDocumentLoader(
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Vehicle {row['vehicle_id']} recorded "
            f"{row['event_count']} events with an average speed "
            f"of {row['avg_speed']}."
        ),
        metadata_fn=lambda row: {
            "source": "gold.vehicle_metrics",
            "data_layer": "gold",
            "dataset": "vehicle_metrics",
            "entity_type": "vehicle",
            "vehicle_id": row["vehicle_id"],
        },
    )

    documents = loader.load(dataframe)

    assert len(documents) == 2

    assert documents[0].id == "vehicle:V001"
    assert documents[0].content == ("Vehicle V001 recorded 3 events with an average speed of 48.5.")
    assert documents[0].metadata == {
        "source": "gold.vehicle_metrics",
        "data_layer": "gold",
        "dataset": "vehicle_metrics",
        "entity_type": "vehicle",
        "vehicle_id": "V001",
    }


def test_spark_dataframe_loader_preserves_multiple_rows(spark):
    dataframe = spark.createDataFrame(
        [
            ("V001", 3),
            ("V002", 4),
            ("V003", 5),
        ],
        ["vehicle_id", "event_count"],
    )

    loader = SparkDataFrameDocumentLoader(
        id_fn=lambda row: f"vehicle:{row['vehicle_id']}",
        content_fn=lambda row: (
            f"Vehicle {row['vehicle_id']} recorded " f"{row['event_count']} events."
        ),
        metadata_fn=lambda row: {
            "vehicle_id": row["vehicle_id"],
        },
    )

    documents = loader.load(dataframe)

    assert [document.id for document in documents] == [
        "vehicle:V001",
        "vehicle:V002",
        "vehicle:V003",
    ]


def test_spark_dataframe_loader_rejects_empty_document_id(spark):
    dataframe = spark.createDataFrame(
        [("", 3)],
        ["vehicle_id", "event_count"],
    )

    loader = SparkDataFrameDocumentLoader(
        id_fn=lambda row: row["vehicle_id"],
        content_fn=lambda row: "vehicle metrics",
        metadata_fn=lambda row: {},
    )

    with pytest.raises(ValueError, match="non-empty document ID"):
        loader.load(dataframe)


def test_spark_dataframe_loader_rejects_empty_content(spark):
    dataframe = spark.createDataFrame(
        [("V001", 3)],
        ["vehicle_id", "event_count"],
    )

    loader = SparkDataFrameDocumentLoader(
        id_fn=lambda row: row["vehicle_id"],
        content_fn=lambda row: "",
        metadata_fn=lambda row: {},
    )

    with pytest.raises(ValueError, match="non-empty text"):
        loader.load(dataframe)


def test_spark_dataframe_loader_supports_recursive_row_values(spark):
    dataframe = spark.createDataFrame(
        [
            ("V001", {"region": "south", "fleet": "A"}),
        ],
        ["vehicle_id", "attributes"],
    )

    loader = SparkDataFrameDocumentLoader(
        id_fn=lambda row: row["vehicle_id"],
        content_fn=lambda row: (
            f"Vehicle {row['vehicle_id']} belongs to " f"fleet {row['attributes']['fleet']}."
        ),
        metadata_fn=lambda row: {
            "vehicle_id": row["vehicle_id"],
            "region": row["attributes"]["region"],
        },
    )

    documents = loader.load(dataframe)

    assert documents[0].content == "Vehicle V001 belongs to fleet A."
    assert documents[0].metadata["region"] == "south"
