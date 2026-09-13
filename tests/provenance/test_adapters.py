from common.databricks.metadata import (
    DatabricksColumnMetadata,
    DatabricksTableMetadata,
)
from common.fabric.metadata import FabricColumnMetadata, FabricTableMetadata
from common.provenance import (
    databricks_source_ref,
    fabric_source_ref,
    snowflake_source_ref,
)
from common.snowflake.metadata import (
    SnowflakeColumnMetadata,
    SnowflakeTableMetadata,
)


def test_databricks_metadata_projects_to_source_ref():
    metadata = DatabricksTableMetadata(
        full_name="vehicle_platform.analytics.rpt_vehicle_summary",
        catalog="vehicle_platform",
        schema="analytics",
        name="rpt_vehicle_summary",
        table_type="MANAGED",
        table_id="table-123",
        columns=(
            DatabricksColumnMetadata(
                name="vehicle_id",
                type_name="STRING",
            ),
        ),
    )

    assert databricks_source_ref(metadata).to_dict() == {
        "platform": "databricks",
        "object_type": "table",
        "object_name": "rpt_vehicle_summary",
        "object_id": "table-123",
        "namespace": "vehicle_platform.analytics",
    }


def test_snowflake_metadata_projects_to_source_ref():
    metadata = SnowflakeTableMetadata(
        full_name="VEHICLE_PLATFORM.ANALYTICS.RPT_VEHICLE_SUMMARY",
        database="VEHICLE_PLATFORM",
        schema="ANALYTICS",
        name="RPT_VEHICLE_SUMMARY",
        table_type="BASE TABLE",
        columns=(
            SnowflakeColumnMetadata(
                name="VEHICLE_ID",
                type_name="VARCHAR",
            ),
        ),
    )

    assert snowflake_source_ref(metadata).to_dict() == {
        "platform": "snowflake",
        "object_type": "table",
        "object_name": "RPT_VEHICLE_SUMMARY",
        "namespace": "VEHICLE_PLATFORM.ANALYTICS",
    }


def test_fabric_metadata_projects_to_source_ref():
    metadata = FabricTableMetadata(
        workspace_id="workspace-123",
        workspace_name="Vehicle Workspace",
        lakehouse_id="lakehouse-123",
        lakehouse_name="Vehicle Lakehouse",
        catalog_name="VehicleLakehouse",
        schema_name="dbo",
        table_id="table-456",
        table_name="vehicle_events",
        table_type="DELTA",
        columns=(
            FabricColumnMetadata(
                name="vehicle_id",
                type_name="string",
            ),
        ),
    )

    assert fabric_source_ref(metadata).to_dict() == {
        "platform": "fabric",
        "object_type": "table",
        "object_name": "vehicle_events",
        "object_id": "table-456",
        "namespace": "Vehicle Workspace.Vehicle Lakehouse.dbo",
    }


def test_databricks_source_ref_can_use_full_name_when_name_missing():
    metadata = DatabricksTableMetadata(
        full_name="vehicle_platform.analytics.vehicle_events",
        catalog="vehicle_platform",
        schema="analytics",
        table_id="table-789",
    )

    assert databricks_source_ref(metadata).to_dict() == {
        "platform": "databricks",
        "object_type": "table",
        "object_name": "vehicle_platform.analytics.vehicle_events",
        "object_id": "table-789",
        "namespace": "vehicle_platform.analytics",
    }


def test_snowflake_source_ref_can_use_full_name_when_name_missing():
    metadata = SnowflakeTableMetadata(
        full_name="VEHICLE_PLATFORM.ANALYTICS.VEHICLE_EVENTS",
        database="VEHICLE_PLATFORM",
        schema="ANALYTICS",
    )

    assert snowflake_source_ref(metadata).to_dict() == {
        "platform": "snowflake",
        "object_type": "table",
        "object_name": "VEHICLE_PLATFORM.ANALYTICS.VEHICLE_EVENTS",
        "namespace": "VEHICLE_PLATFORM.ANALYTICS",
    }


def test_fabric_source_ref_requires_table_identity():
    metadata = FabricTableMetadata(
        workspace_name="Vehicle Workspace",
        lakehouse_name="Vehicle Lakehouse",
        schema_name="dbo",
        table_name="vehicle_events",
    )

    assert fabric_source_ref(metadata).to_dict() == {
        "platform": "fabric",
        "object_type": "table",
        "object_name": "vehicle_events",
        "namespace": "Vehicle Workspace.Vehicle Lakehouse.dbo",
    }


def test_s3_path_projects_to_source_ref():
    from common.provenance import s3_source_ref

    assert s3_source_ref("s3://enterprise-data-ai-platform/bronze/vehicle_events").to_dict() == {
        "platform": "aws",
        "object_type": "s3_path",
        "object_name": "s3://enterprise-data-ai-platform/bronze/vehicle_events",
        "namespace": "s3://enterprise-data-ai-platform",
    }


def test_s3_path_normalizes_trailing_slash():
    from common.provenance import s3_source_ref

    source_ref = s3_source_ref("s3://enterprise-data-ai-platform/bronze/vehicle_events/")

    assert source_ref.object_name == ("s3://enterprise-data-ai-platform/bronze/vehicle_events")


def test_s3_path_rejects_non_s3_uri():
    import pytest

    from common.provenance import s3_source_ref

    with pytest.raises(ValueError, match="s3://"):
        s3_source_ref("https://example.com/vehicle_events")


def test_s3_path_rejects_bucket_only_uri():
    import pytest

    from common.provenance import s3_source_ref

    with pytest.raises(ValueError, match="bucket and object path"):
        s3_source_ref("s3://enterprise-data-ai-platform")


def test_s3_path_rejects_empty_path():
    import pytest

    from common.provenance import s3_source_ref

    with pytest.raises(ValueError, match="cannot be empty"):
        s3_source_ref("   ")
