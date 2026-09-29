from common.aws.catalog_synchronizer import AwsGlueCatalogSynchronizer
from common.aws.pipeline_catalog import build_aws_glue_synchronizer


class FakeGlueClient:
    pass


def test_build_aws_glue_synchronizer_uses_requested_region(monkeypatch):
    client = FakeGlueClient()
    calls = []

    def fake_client(service_name, region_name=None):
        calls.append(
            {
                "service_name": service_name,
                "region_name": region_name,
            }
        )
        return client

    monkeypatch.setattr(
        "common.aws.pipeline_catalog.boto3.client",
        fake_client,
    )

    synchronizer = build_aws_glue_synchronizer(
        region_name="us-east-1",
    )

    assert isinstance(synchronizer, AwsGlueCatalogSynchronizer)
    assert calls == [
        {
            "service_name": "glue",
            "region_name": "us-east-1",
        }
    ]


def test_build_aws_glue_synchronizer_uses_aws_region(monkeypatch):
    client = FakeGlueClient()
    calls = []

    monkeypatch.setenv("AWS_REGION", "us-east-1")

    def fake_client(service_name, region_name=None):
        calls.append(
            {
                "service_name": service_name,
                "region_name": region_name,
            }
        )
        return client

    monkeypatch.setattr(
        "common.aws.pipeline_catalog.boto3.client",
        fake_client,
    )

    synchronizer = build_aws_glue_synchronizer()

    assert isinstance(synchronizer, AwsGlueCatalogSynchronizer)
    assert calls == [
        {
            "service_name": "glue",
            "region_name": "us-east-1",
        }
    ]
