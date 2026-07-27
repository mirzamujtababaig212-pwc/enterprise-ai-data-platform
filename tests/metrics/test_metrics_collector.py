from unittest.mock import patch

from common.metrics.metrics_collector import MetricsCollector


def test_create():
    collector = MetricsCollector()
    assert collector is not None

def test_record_batch(spark):
    collector = MetricsCollector()
    batch = spark.createDataFrame(
        [
            (1,"A"),
            (2,"B")
        ],
        ["id","name"]
    )
    rejected = spark.createDataFrame(
        [],
        batch.schema
    )
    collector.record_batch(
        pipeline="bronze",
        batch_id=1,
        batch_df=batch,
        rejected_df=rejected
    )

@patch(
    "common.metrics.metrics_collector.logger"
)

def test_logger_called(
    mock_logger,
    spark
):
    collector = MetricsCollector()
    batch = spark.createDataFrame(
        [(1,"A")],
        ["id","name"]
    )
    rejected = spark.createDataFrame(
        [],
        batch.schema
    )
    collector.record_batch(
        pipeline="bronze",
        batch_id=1,
        batch_df=batch,
        rejected_df=rejected
    )
    assert mock_logger.info.called

@patch("common.metrics.metrics_collector.logger")
def test_row_count(mock_logger, spark):
    collector = MetricsCollector()
    batch = spark.createDataFrame(
        [
            (1, "A"),
            (2, "B"),
        ],
        ["id", "name"],
    )
    rejected = spark.createDataFrame(
        [
            (3, "C"),
        ],
        batch.schema,
    )
    metrics = collector.record_batch(
        pipeline="bronze",
        batch_id=1,
        batch_df=batch,
        rejected_df=rejected,
    )
    assert metrics["processed"]==2
    assert metrics["rejected"]==1

@patch("common.metrics.metrics_collector.logger")
def test_batch_id(mock_logger, spark):
    collector = MetricsCollector()
    batch = spark.createDataFrame(
        [(1, "A")],
        ["id", "name"],
    )
    collector.record_batch(
        pipeline="bronze",
        batch_id=10,
        batch_df=batch,
    )
    log_text = str(mock_logger.info.call_args_list)
    assert "10" in log_text

@patch("common.metrics.metrics_collector.logger")
def test_pipeline_name(mock_logger, spark):
    collector = MetricsCollector()
    batch = spark.createDataFrame(
        [(1, "A")],
        ["id", "name"],
    )
    collector.record_batch(
        pipeline="silver",
        batch_id=1,
        batch_df=batch,
    )
    assert "silver" in str(mock_logger.info.call_args_list)

@patch("common.metrics.metrics_collector.logger")
def test_rejected_rows(mock_logger, spark):
    collector = MetricsCollector()
    batch = spark.createDataFrame(
        [(i, f"name{i}") for i in range(10)],
        ["id", "name"],
    )
    rejected = spark.createDataFrame(
        [
            (1, "bad"),
            (2, "bad"),
            (3, "bad"),
        ],
        ["id", "name"],
    )
    collector.record_batch(
        pipeline="bronze",
        batch_id=1,
        batch_df=batch,
        rejected_df=rejected,
    )
    assert mock_logger.info.called

def test_empty_batch(spark):
    collector = MetricsCollector()
    batch = spark.createDataFrame(
        [],
        "id INT, name STRING",
    )
    rejected = spark.createDataFrame(
        [],
        batch.schema,
    )
    collector.record_batch(
        pipeline="bronze",
        batch_id=1,
        batch_df=batch,
        rejected_df=rejected,
    )
    assert batch.count() == 0

def test_large_batch(spark):
    collector = MetricsCollector()
    rows = [
        (i, f"name{i}")
        for i in range(10000)
    ]
    batch = spark.createDataFrame(
        rows,
        ["id", "name"],
    )
    rejected = spark.createDataFrame(
        [],
        batch.schema,
    )
    collector.record_batch(
        pipeline="bronze",
        batch_id=1,
        batch_df=batch,
        rejected_df=rejected,
    )
    assert batch.count() == 10000

@patch("common.metrics.metrics_collector.time")
def test_duration(mock_time, spark):
    mock_time.time.side_effect = [
        100,
        103,
    ]
    collector = MetricsCollector()
    batch = spark.createDataFrame(
        [(1, "A")],
        ["id", "name"],
    )
    rejected = spark.createDataFrame([], batch.schema)
    collector.record_batch(
        pipeline="bronze",
        batch_id=1,
        batch_df=batch,
        rejected_df=rejected,
    )

