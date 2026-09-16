from __future__ import annotations

import gzip
import re
import tarfile
from pathlib import Path
from typing import Any

import mlflow
import skops.io as sio
from mlflow import MlflowClient

from common.storage.s3_storage import S3Storage

MODEL_FILENAME = "model.skops"
REQUIRED_ARTIFACTS = (
    "MLmodel",
    "model.skops",
    "requirements.txt",
)

MODEL_SOURCE_PATTERN = re.compile(r"models:/(?P<model_id>[^/]+)")


def get_champion_model_version(
    model_name: str,
    model_alias: str = "champion",
    client: MlflowClient | None = None,
) -> Any:
    """Resolve a registered-model alias to its MLflow model version."""
    client = client or MlflowClient()

    version = client.get_model_version_by_alias(
        model_name,
        model_alias,
    )

    if version.name != model_name:
        raise ValueError(
            f"Resolved model name {version.name!r} does not match "
            f"requested model {model_name!r}"
        )

    return version


def extract_model_id(source: str) -> str:
    """Extract the MLflow logged-model ID from a models:/ source URI."""
    match = MODEL_SOURCE_PATTERN.fullmatch(source)

    if not match:
        raise ValueError(
            f"Unsupported MLflow model source: {source!r}. " "Expected models:/<model-id>."
        )

    return match.group("model_id")


def download_champion_artifact(
    model_name: str,
    model_alias: str = "champion",
    client: MlflowClient | None = None,
) -> tuple[Any, Path]:
    """Resolve and download the MLflow champion logged-model artifact."""
    version = get_champion_model_version(
        model_name=model_name,
        model_alias=model_alias,
        client=client,
    )

    artifact_dir = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=version.source,
        )
    )

    validate_model_artifact(artifact_dir)

    return version, artifact_dir


def validate_model_artifact(artifact_dir: str | Path) -> None:
    """Validate the files and SKOPS trust boundary of a model artifact."""
    artifact_dir = Path(artifact_dir)

    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"Model artifact directory not found: {artifact_dir}")

    missing = [
        filename for filename in REQUIRED_ARTIFACTS if not (artifact_dir / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError("Model artifact is missing required files: " + ", ".join(missing))

    model_path = artifact_dir / MODEL_FILENAME
    untrusted_types = sio.get_untrusted_types(file=str(model_path))

    if untrusted_types:
        raise ValueError(
            "Model artifact contains untrusted SKOPS types: " + ", ".join(untrusted_types)
        )


def package_model_artifact(
    artifact_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Create a deterministic SageMaker model.tar.gz from an MLflow artifact."""
    artifact_dir = Path(artifact_dir)
    output_path = Path(output_path)

    validate_model_artifact(artifact_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    files = [artifact_dir / filename for filename in REQUIRED_ARTIFACTS]

    with output_path.open("wb") as raw_file:
        with gzip.GzipFile(
            fileobj=raw_file,
            mode="wb",
            filename="",
            mtime=0,
        ) as gzip_file:
            with tarfile.open(
                fileobj=gzip_file,
                mode="w",
            ) as archive:
                for path in files:
                    info = archive.gettarinfo(
                        str(path),
                        arcname=path.name,
                    )
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0

                    with path.open("rb") as source:
                        archive.addfile(info, source)

    return output_path


def champion_artifact_key(
    model_name: str,
    model_version: str,
    model_id: str,
) -> str:
    """Build the immutable S3 object key for a packaged model artifact."""
    return (
        f"model-artifacts/{model_name.lower()}/"
        f"model-version-{model_version}/{model_id}/model.tar.gz"
    )


def package_champion_model(
    model_name: str,
    output_path: str | Path,
    model_alias: str = "champion",
    client: MlflowClient | None = None,
) -> tuple[Any, Path]:
    """Resolve, validate, download, and package an MLflow champion."""
    version, artifact_dir = download_champion_artifact(
        model_name=model_name,
        model_alias=model_alias,
        client=client,
    )

    package_path = package_model_artifact(
        artifact_dir=artifact_dir,
        output_path=output_path,
    )

    return version, package_path


def publish_champion_model(
    model_name: str,
    output_path: str | Path,
    storage: S3Storage,
    model_alias: str = "champion",
    client: MlflowClient | None = None,
) -> tuple[Any, Path, str, str]:
    """Package and publish the MLflow champion model to S3."""
    version, package_path = package_champion_model(
        model_name=model_name,
        output_path=output_path,
        model_alias=model_alias,
        client=client,
    )

    model_id = extract_model_id(version.source)
    key = champion_artifact_key(
        model_name=model_name,
        model_version=str(version.version),
        model_id=model_id,
    )

    if storage.exists(key):
        raise FileExistsError(f"Immutable model artifact already exists: {storage.uri(key)}")

    storage.write(key, package_path.read_bytes())

    return version, package_path, key, storage.uri(key)
