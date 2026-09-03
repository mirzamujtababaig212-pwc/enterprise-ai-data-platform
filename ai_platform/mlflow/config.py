from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    s3_endpoint_url: str | None
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_default_region: str

    @classmethod
    def from_environment(
        cls,
        experiment_name: str = "enterprise-ai-platform",
    ) -> "MLflowConfig":
        return cls(
            tracking_uri=os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://mlflow:5000",
            ),
            experiment_name=experiment_name,
            s3_endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_default_region=os.getenv(
                "AWS_DEFAULT_REGION",
                "us-east-1",
            ),
        )

    def validate(self) -> None:
        if not self.tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI is not configured")

        if not self.experiment_name:
            raise ValueError("MLflow experiment name cannot be empty")

        if self.s3_endpoint_url and not self.aws_access_key_id:
            raise ValueError(
                "AWS_ACCESS_KEY_ID is required when " "MLFLOW_S3_ENDPOINT_URL is configured"
            )

        if self.s3_endpoint_url and not self.aws_secret_access_key:
            raise ValueError(
                "AWS_SECRET_ACCESS_KEY is required when " "MLFLOW_S3_ENDPOINT_URL is configured"
            )
