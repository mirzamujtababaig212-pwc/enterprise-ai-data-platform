from __future__ import annotations

import gzip
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier

from ml.sagemaker.artifacts import (
    champion_artifact_key,
    extract_model_id,
    publish_champion_model,
    package_model_artifact,
    validate_model_artifact,
)


def _write_artifact(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(
        [[1.0], [2.0], [3.0], [4.0]],
        [0, 0, 1, 1],
    )

    sio.dump(model, str(directory / "model.skops"))
    (directory / "MLmodel").write_text("test-model")
    (directory / "requirements.txt").write_text("skops==0.14.0\n")

    return directory


def test_extract_model_id() -> None:
    assert (
        extract_model_id("models:/m-f68ad14c07064a1a9f2619fa7a7d9d7a")
        == "m-f68ad14c07064a1a9f2619fa7a7d9d7a"
    )


def test_extract_model_id_rejects_invalid_source() -> None:
    with pytest.raises(ValueError, match="Unsupported MLflow model source"):
        extract_model_id("runs:/123/model")


def test_validate_model_artifact_accepts_valid_artifact(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(tmp_path)

    validate_model_artifact(artifact_dir)


def test_validate_model_artifact_rejects_missing_file(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="missing required files"):
        validate_model_artifact(artifact_dir)


def test_validate_model_artifact_rejects_untrusted_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = _write_artifact(tmp_path)

    monkeypatch.setattr(
        sio,
        "get_untrusted_types",
        lambda **kwargs: ["evil.UntrustedType"],
    )

    with pytest.raises(ValueError, match="untrusted SKOPS types"):
        validate_model_artifact(artifact_dir)


def test_package_model_artifact_creates_expected_archive(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(tmp_path / "artifact")
    output_path = tmp_path / "model.tar.gz"

    package_model_artifact(
        artifact_dir=artifact_dir,
        output_path=output_path,
    )

    assert output_path.is_file()

    with gzip.open(output_path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r") as archive:
            assert archive.getnames() == [
                "MLmodel",
                "model.skops",
                "requirements.txt",
            ]


def test_package_model_artifact_excludes_non_deployment_files(
    tmp_path: Path,
) -> None:
    artifact_dir = _write_artifact(tmp_path / "artifact")
    (artifact_dir / "conda.yaml").write_text("name: ignored")
    (artifact_dir / "python_env.yaml").write_text("ignored: true")
    (artifact_dir / "registered_model_meta").write_text("ignored")

    output_path = tmp_path / "model.tar.gz"

    package_model_artifact(
        artifact_dir=artifact_dir,
        output_path=output_path,
    )

    with gzip.open(output_path, "rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r") as archive:
            assert archive.getnames() == [
                "MLmodel",
                "model.skops",
                "requirements.txt",
            ]


def test_package_model_artifact_is_deterministic(tmp_path: Path) -> None:
    artifact_dir = _write_artifact(tmp_path / "artifact")

    first = package_model_artifact(
        artifact_dir=artifact_dir,
        output_path=tmp_path / "first.tar.gz",
    ).read_bytes()

    second = package_model_artifact(
        artifact_dir=artifact_dir,
        output_path=tmp_path / "second.tar.gz",
    ).read_bytes()

    assert first == second


def test_champion_artifact_key() -> None:
    assert champion_artifact_key(
        model_name="VehicleRiskModel",
        model_version="2",
        model_id="m-f68ad14c07064a1a9f2619fa7a7d9d7a",
    ) == (
        "model-artifacts/vehicleriskmodel/"
        "model-version-2/m-f68ad14c07064a1a9f2619fa7a7d9d7a/model.tar.gz"
    )


def test_publish_champion_model_uploads_immutable_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = tmp_path / "model.tar.gz"
    package_path.write_bytes(b"deployment-artifact")

    version = MagicMock()
    version.source = "models:/m-f68ad14c07064a1a9f2619fa7a7d9d7a"
    version.version = "2"

    package_mock = MagicMock(return_value=(version, package_path))
    monkeypatch.setattr(
        "ml.sagemaker.artifacts.package_champion_model",
        package_mock,
    )

    storage = MagicMock()
    storage.exists.return_value = False
    storage.uri.return_value = (
        "s3://enterprise-data-ai-platform/"
        "model-artifacts/vehicleriskmodel/"
        "model-version-2/"
        "m-f68ad14c07064a1a9f2619fa7a7d9d7a/model.tar.gz"
    )

    result = publish_champion_model(
        model_name="VehicleRiskModel",
        output_path=tmp_path / "model.tar.gz",
        storage=storage,
    )

    assert result == (
        version,
        package_path,
        "model-artifacts/vehicleriskmodel/"
        "model-version-2/m-f68ad14c07064a1a9f2619fa7a7d9d7a/model.tar.gz",
        "s3://enterprise-data-ai-platform/"
        "model-artifacts/vehicleriskmodel/"
        "model-version-2/m-f68ad14c07064a1a9f2619fa7a7d9d7a/model.tar.gz",
    )

    package_mock.assert_called_once_with(
        model_name="VehicleRiskModel",
        output_path=tmp_path / "model.tar.gz",
        model_alias="champion",
        client=None,
    )
    storage.exists.assert_called_once_with(
        "model-artifacts/vehicleriskmodel/"
        "model-version-2/m-f68ad14c07064a1a9f2619fa7a7d9d7a/model.tar.gz"
    )
    storage.uri.assert_called()
    storage.write.assert_called_once_with(
        "model-artifacts/vehicleriskmodel/"
        "model-version-2/m-f68ad14c07064a1a9f2619fa7a7d9d7a/model.tar.gz",
        b"deployment-artifact",
    )


def test_publish_champion_model_rejects_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_path = tmp_path / "model.tar.gz"
    package_path.write_bytes(b"deployment-artifact")

    version = MagicMock()
    version.source = "models:/m-f68ad14c07064a1a9f2619fa7a7d9d7a"
    version.version = "2"

    monkeypatch.setattr(
        "ml.sagemaker.artifacts.package_champion_model",
        MagicMock(return_value=(version, package_path)),
    )

    storage = MagicMock()
    storage.exists.return_value = True
    storage.uri.return_value = (
        "s3://enterprise-data-ai-platform/model-artifacts/"
        "vehicleriskmodel/model-version-2/"
        "m-f68ad14c07064a1a9f2619fa7a7d9d7a/model.tar.gz"
    )

    with pytest.raises(FileExistsError, match="Immutable model artifact already exists"):
        publish_champion_model(
            model_name="VehicleRiskModel",
            output_path=package_path,
            storage=storage,
        )

    storage.write.assert_not_called()
