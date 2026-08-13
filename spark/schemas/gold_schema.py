from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

gold_schema = StructType(
    [
        StructField(
            "vehicle_id",
            StringType(),
            nullable=False,
        ),
        StructField(
            "event_count",
            LongType(),
            nullable=False,
        ),
        StructField(
            "avg_speed",
            DoubleType(),
            nullable=False,
        ),
        StructField(
            "min_speed",
            DoubleType(),
            nullable=False,
        ),
        StructField(
            "max_speed",
            DoubleType(),
            nullable=False,
        ),
        StructField(
            "first_event_time",
            TimestampType(),
            nullable=False,
        ),
        StructField(
            "last_event_time",
            TimestampType(),
            nullable=False,
        ),
    ]
)


GOLD_REQUIRED_COLUMNS = [
    "vehicle_id",
    "event_count",
    "avg_speed",
    "min_speed",
    "max_speed",
    "first_event_time",
    "last_event_time",
]
