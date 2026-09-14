from common.aws.metadata import (
    AwsGlueColumnMetadata,
    AwsGlueTableMetadata,
)
from common.provenance import aws_glue_source_ref


def test_aws_glue_metadata_projects_to_source_ref():
    metadata = AwsGlueTableMetadata(
        catalog_id="715621342004",
        database_name="enterprise_ai",
        name="vehicle_events",
        table_type="EXTERNAL_TABLE",
        location="s3://enterprise-data-ai-platform/bronze/vehicle_events",
        columns=(
            AwsGlueColumnMetadata(
                name="vehicle_id",
                type_name="string",
            ),
        ),
    )

    assert aws_glue_source_ref(metadata).to_dict() == {
        "platform": "aws",
        "object_type": "table",
        "object_name": "vehicle_events",
        "namespace": "enterprise_ai",
    }


def test_aws_glue_source_ref_can_work_without_catalog_id():
    metadata = AwsGlueTableMetadata(
        database_name="bronze",
        name="vehicle_events",
    )

    assert aws_glue_source_ref(metadata).to_dict() == {
        "platform": "aws",
        "object_type": "table",
        "object_name": "vehicle_events",
        "namespace": "bronze",
    }
