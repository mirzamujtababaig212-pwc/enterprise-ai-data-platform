from common.snowflake.metadata import (
    SnowflakeColumnMetadata,
    SnowflakeTableMetadata,
)


def test_snowflake_column_metadata_defaults():
    metadata = SnowflakeColumnMetadata()

    assert metadata.name is None
    assert metadata.type_name is None
    assert metadata.type_text is None
    assert metadata.nullable is None
    assert metadata.comment is None
    assert metadata.position is None


def test_snowflake_table_metadata_defaults():
    metadata = SnowflakeTableMetadata()

    assert metadata.full_name is None
    assert metadata.database is None
    assert metadata.schema is None
    assert metadata.name is None
    assert metadata.table_type is None
    assert metadata.owner is None
    assert metadata.comment is None
    assert metadata.columns == ()


def test_snowflake_metadata_preserves_table_and_column_values():
    column_a = SnowflakeColumnMetadata(
        name="vehicle_id",
        type_name="NUMBER",
        type_text="NUMBER(38,0)",
        nullable=False,
        comment="Vehicle identifier",
        position=1,
    )
    column_b = SnowflakeColumnMetadata(
        name="avg_speed",
        type_name="FLOAT",
        type_text="FLOAT",
        nullable=True,
        comment="Average speed",
        position=2,
    )

    metadata = SnowflakeTableMetadata(
        full_name="VEHICLE_PLATFORM.ANALYTICS.RPT_VEHICLE_SUMMARY",
        database="VEHICLE_PLATFORM",
        schema="ANALYTICS",
        name="RPT_VEHICLE_SUMMARY",
        table_type="BASE TABLE",
        owner="DATA_PLATFORM",
        comment="Vehicle fleet summary",
        columns=(column_a, column_b),
    )

    assert metadata.full_name == "VEHICLE_PLATFORM.ANALYTICS.RPT_VEHICLE_SUMMARY"
    assert metadata.database == "VEHICLE_PLATFORM"
    assert metadata.schema == "ANALYTICS"
    assert metadata.name == "RPT_VEHICLE_SUMMARY"
    assert metadata.table_type == "BASE TABLE"
    assert metadata.owner == "DATA_PLATFORM"
    assert metadata.comment == "Vehicle fleet summary"

    assert len(metadata.columns) == 2
    assert metadata.columns[0].name == "vehicle_id"
    assert metadata.columns[0].type_name == "NUMBER"
    assert metadata.columns[0].type_text == "NUMBER(38,0)"
    assert metadata.columns[0].nullable is False
    assert metadata.columns[0].position == 1

    assert metadata.columns[1].name == "avg_speed"
    assert metadata.columns[1].type_name == "FLOAT"
    assert metadata.columns[1].nullable is True
    assert metadata.columns[1].position == 2
