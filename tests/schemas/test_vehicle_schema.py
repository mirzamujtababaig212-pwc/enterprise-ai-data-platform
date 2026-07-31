from pyspark.sql.types import StructType

from spark.schemas.vehicle_schema import vehicle_schema


def test_vehicle_schema_type():
    assert isinstance(vehicle_schema, StructType)


def test_vehicle_schema_field_count():
    assert len(vehicle_schema.fields) == 10


def test_vehicle_schema_first_field():
    assert vehicle_schema.fields[0].name == "vehicle_id"


def test_vehicle_schema_last_field():
    assert vehicle_schema.fields[-1].name == "gear"
