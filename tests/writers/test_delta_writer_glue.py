from unittest.mock import Mock

import pytest

from common.writers.delta_writer import DeltaWriter


class FakeDataFrameWriter:
    def __init__(self, spark_session):
        self.spark_session = spark_session
        self.saved = False

    def format(self, value):
        assert value == "delta"
        return self

    def mode(self, value):
        assert value == "append"
        return self

    def option(self, key, value):
        assert key == "overwriteSchema"
        assert value == "true"
        return self

    def save(self, path):
        self.saved = True
        self.saved_path = path


class FakeDataFrame:
    def __init__(self, spark_session, schema):
        self.sparkSession = spark_session
        self.schema = schema
        self.write = FakeDataFrameWriter(spark_session)


class FakeSparkCatalog:
    def tableExists(self, table):
        return False


class FakeSpark:
    def __init__(self):
        self.catalog = FakeSparkCatalog()

    def sql(self, statement):
        return Mock()


class FakeGlueSynchronizer:
    def __init__(self):
        self.calls = []

    def sync(self, **kwargs):
        self.calls.append(kwargs)
        return "created"


class FailingDataFrameWriter(FakeDataFrameWriter):
    def save(self, path):
        raise RuntimeError("delta write failed")


class FailingDataFrame(FakeDataFrame):
    def __init__(self, spark_session, schema):
        super().__init__(spark_session, schema)
        self.write = FailingDataFrameWriter(spark_session)


def _schema():
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    return StructType(
        [
            StructField("vehicle_id", StringType(), True),
            StructField("event_time", TimestampType(), True),
            StructField("speed", DoubleType(), True),
        ]
    )


def _writer(glue_synchronizer=None):
    return DeltaWriter(
        table="bronze.vehicle_events",
        path="s3a://enterprise-data-ai-platform/bronze/vehicle_events",
        mode="append",
        glue_synchronizer=glue_synchronizer,
        glue_database_name="enterprise_ai_platform",
        glue_table_name="vehicle_events",
    )


def test_delta_write_does_not_sync_glue_when_disabled():
    spark = FakeSpark()
    df = FakeDataFrame(spark, _schema())

    writer = DeltaWriter(
        table="bronze.vehicle_events",
        path="s3a://enterprise-data-ai-platform/bronze/vehicle_events",
        mode="append",
    )

    writer._register_table = Mock()

    writer.write_batch(df)

    writer._register_table.assert_called_once_with(spark)


def test_delta_write_syncs_glue_after_successful_write():
    spark = FakeSpark()
    df = FakeDataFrame(spark, _schema())

    synchronizer = FakeGlueSynchronizer()

    writer = _writer(synchronizer)
    writer._register_table = Mock()

    writer.write_batch(df)

    assert len(synchronizer.calls) == 1

    call = synchronizer.calls[0]

    assert call["database_name"] == "enterprise_ai_platform"

    table_input = call["table_input"]

    assert table_input["Name"] == "vehicle_events"
    assert table_input["TableType"] == "EXTERNAL_TABLE"
    assert table_input["StorageDescriptor"]["Location"] == (
        "s3://enterprise-data-ai-platform/bronze/vehicle_events"
    )
    assert table_input["StorageDescriptor"]["Columns"] == [
        {"Name": "vehicle_id", "Type": "string"},
        {"Name": "event_time", "Type": "timestamp"},
        {"Name": "speed", "Type": "double"},
    ]


def test_delta_write_does_not_sync_glue_when_delta_write_fails():
    spark = FakeSpark()
    df = FailingDataFrame(spark, _schema())

    synchronizer = FakeGlueSynchronizer()

    writer = _writer(synchronizer)
    writer._register_table = Mock()

    with pytest.raises(RuntimeError, match="delta write failed"):
        writer.write_batch(df)

    assert synchronizer.calls == []
    writer._register_table.assert_not_called()
