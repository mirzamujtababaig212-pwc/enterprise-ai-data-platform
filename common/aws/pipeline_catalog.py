from __future__ import annotations

import os

import boto3

from common.aws.catalog_publisher import AwsGlueCatalogPublisher
from common.aws.catalog_synchronizer import AwsGlueCatalogSynchronizer
from common.aws.control_plane import AwsGlueControlPlaneClient


def build_aws_glue_synchronizer(
    *,
    region_name: str | None = None,
) -> AwsGlueCatalogSynchronizer:
    """
    Build the AWS Glue catalog synchronizer at the AWS execution boundary.

    The boto3 Glue client is created once and shared by the read-only
    control-plane client and mutation publisher.
    """
    resolved_region = region_name or os.getenv("AWS_REGION") or None

    client = boto3.client(
        "glue",
        region_name=resolved_region,
    )

    return AwsGlueCatalogSynchronizer(
        control_plane=AwsGlueControlPlaneClient(client=client),
        publisher=AwsGlueCatalogPublisher(client=client),
    )
