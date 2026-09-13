import pytest

from common.provenance.source import EnterpriseSourceRef


def test_source_ref_preserves_identity_fields():
    source_ref = EnterpriseSourceRef(
        platform="databricks",
        object_type="table",
        object_name="rpt_vehicle_summary",
        object_id="table-123",
        namespace="vehicle_platform.analytics",
        environment="dev",
        version="v1",
    )

    assert source_ref.platform == "databricks"
    assert source_ref.object_type == "table"
    assert source_ref.object_name == "rpt_vehicle_summary"
    assert source_ref.object_id == "table-123"
    assert source_ref.namespace == "vehicle_platform.analytics"
    assert source_ref.environment == "dev"
    assert source_ref.version == "v1"


def test_source_ref_to_dict_omits_none_values():
    source_ref = EnterpriseSourceRef(
        platform="snowflake",
        object_type="table",
        object_name="RPT_VEHICLE_SUMMARY",
        namespace="VEHICLE_PLATFORM.ANALYTICS",
    )

    assert source_ref.to_dict() == {
        "platform": "snowflake",
        "object_type": "table",
        "object_name": "RPT_VEHICLE_SUMMARY",
        "namespace": "VEHICLE_PLATFORM.ANALYTICS",
    }


def test_source_ref_accepts_object_id_without_object_name():
    source_ref = EnterpriseSourceRef(
        platform="fabric",
        object_type="table",
        object_id="table-123",
    )

    assert source_ref.to_dict() == {
        "platform": "fabric",
        "object_type": "table",
        "object_id": "table-123",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("platform", ""),
        ("platform", "   "),
        ("object_type", ""),
        ("object_type", "   "),
    ],
)
def test_source_ref_rejects_empty_required_fields(field, value):
    kwargs = {
        "platform": "databricks",
        "object_type": "table",
        "object_name": "vehicle_events",
    }
    kwargs[field] = value

    with pytest.raises(ValueError):
        EnterpriseSourceRef(**kwargs)


def test_source_ref_requires_object_name_or_object_id():
    with pytest.raises(
        ValueError,
        match="requires object_name or object_id",
    ):
        EnterpriseSourceRef(
            platform="databricks",
            object_type="table",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("object_name", ""),
        ("object_name", "   "),
        ("object_id", ""),
        ("object_id", "   "),
    ],
)
def test_source_ref_rejects_empty_optional_identity(field, value):
    kwargs = {
        "platform": "databricks",
        "object_type": "table",
        "object_name": "vehicle_events",
    }
    kwargs[field] = value

    if field == "object_id":
        kwargs["object_name"] = None

    with pytest.raises(ValueError):
        EnterpriseSourceRef(**kwargs)


def test_source_ref_is_immutable():
    source_ref = EnterpriseSourceRef(
        platform="databricks",
        object_type="table",
        object_name="vehicle_events",
    )

    with pytest.raises(AttributeError):
        source_ref.platform = "snowflake"
