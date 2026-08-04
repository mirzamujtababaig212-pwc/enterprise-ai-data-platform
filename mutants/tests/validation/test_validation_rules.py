from pyspark.sql.column import Column

from common.validation.validation_rules import ValidationRules


def test_required_columns():
    required = ValidationRules.required_columns()
    assert required == [
        "vehicle_id",
        "timestamp",
        "speed",
        "fuel_level",
    ]


def test_valid_speed(spark):
    expr = ValidationRules.valid_speed()
    assert isinstance(expr, Column)


def test_valid_fuel(spark):
    expr = ValidationRules.valid_fuel()
    assert isinstance(expr, Column)


def test_valid_engine_temperature(spark):
    expr = ValidationRules.valid_engine_temperature()
    assert isinstance(expr, Column)
