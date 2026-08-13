from pyspark.sql.types import StructType

from spark.schemas.gold_schema import gold_schema


def test_gold_schema_type():
    assert isinstance(gold_schema, StructType)


def test_gold_schema_field_count():
    assert len(gold_schema.fields) == 7


def test_gold_first_field():
    assert gold_schema.fields[0].name == "vehicle_id"


def test_gold_last_field():
    assert gold_schema.fields[-1].name == "last_event_time"
