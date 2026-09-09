from __future__ import annotations

from datetime import datetime

from common.readers.delta_reader import DeltaReader
from rag.loaders import GoldVehicleMetricsDocumentLoader


def test_gold_vehicle_metrics_loader_reads_delta_and_creates_documents(
    spark,
    tmp_path,
):
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
                2,
                34.73,
                32.1,
                36.4,
                datetime(2026, 1, 2, 10, 0, 0),
                datetime(2026, 1, 2, 11, 0, 0),
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

    dataframe.write.format("delta").mode("overwrite").save(str(tmp_path / "gold_vehicle_metrics"))

    reader = DeltaReader(path=str(tmp_path / "gold_vehicle_metrics"))

    loader = GoldVehicleMetricsDocumentLoader(
        reader=reader,
    )

    documents = loader.load(spark)

    assert len(documents) == 2

    assert {document.id for document in documents} == {
        "vehicle:V001",
        "vehicle:V002",
    }

    documents_by_id = {document.id: document for document in documents}

    vehicle_v001 = documents_by_id["vehicle:V001"]

    assert vehicle_v001.metadata["source"] == "gold.vehicle_metrics"
    assert vehicle_v001.metadata["data_layer"] == "gold"
    assert vehicle_v001.metadata["dataset"] == "vehicle_metrics"

    assert "Vehicle V001 recorded 3 events." in vehicle_v001.content
    assert "Average speed was 48.17." in vehicle_v001.content
