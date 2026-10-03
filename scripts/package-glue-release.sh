#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AWS_REGION="${AWS_REGION:-us-east-1}"
ARTIFACT_BUCKET="${GLUE_ARTIFACT_BUCKET:-enterprise-ai-platform-dev-$(aws sts get-caller-identity \
  --query Account \
  --output text)}"

COMMIT_SHA="${GLUE_RELEASE_VERSION:-$(git -C "${ROOT_DIR}" rev-parse HEAD)}"

RELEASE_DIR="${ROOT_DIR}/build/glue-release/${COMMIT_SHA}"

echo "Enterprise AI Platform - Glue Release"
echo "====================================="
echo "AWS region:     ${AWS_REGION}"
echo "Artifact bucket: ${ARTIFACT_BUCKET}"
echo "Release version: ${COMMIT_SHA}"
echo "Release dir:     ${RELEASE_DIR}"
echo

rm -rf "${RELEASE_DIR}"

mkdir -p \
  "${RELEASE_DIR}/scripts" \
  "${RELEASE_DIR}/wheels" \
  "${RELEASE_DIR}/config/environments" \
  "${RELEASE_DIR}/config/pipelines"

echo "Building Python wheel..."

python -m build \
  --wheel \
  --outdir "${RELEASE_DIR}/wheels" \
  "${ROOT_DIR}"

echo "Copying Glue job..."

cp \
  "${ROOT_DIR}/spark/glue/pipeline_job.py" \
  "${RELEASE_DIR}/scripts/pipeline_job.py"

echo "Copying AWS environment configuration..."

cp \
  "${ROOT_DIR}/config/environments/aws.yaml" \
  "${RELEASE_DIR}/config/environments/aws.yaml"

echo "Copying pipeline configurations..."

for pipeline in \
  bronze.yaml \
  silver.yaml \
  gold.yaml \
  silver_streaming.yaml
do
  cp \
    "${ROOT_DIR}/config/pipelines/${pipeline}" \
    "${RELEASE_DIR}/config/pipelines/${pipeline}"
done

WHEEL_PATH="$(find "${RELEASE_DIR}/wheels" -maxdepth 1 -type f -name '*.whl' -print -quit)"

if [[ -z "${WHEEL_PATH}" ]]; then
  echo "ERROR: Glue wheel was not produced." >&2
  exit 1
fi

WHEEL_NAME="$(basename "${WHEEL_PATH}")"

echo "Generating release manifest..."

python - "${RELEASE_DIR}" "${COMMIT_SHA}" "${WHEEL_NAME}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

release_dir = Path(sys.argv[1])
commit_sha = sys.argv[2]
wheel_name = sys.argv[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


files = []

for path in sorted(release_dir.rglob("*")):
    if not path.is_file():
        continue

    files.append(
        {
            "path": path.relative_to(release_dir).as_posix(),
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
    )

manifest = {
    "release_version": commit_sha,
    "wheel": wheel_name,
    "files": files,
}

(release_dir / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo
echo "Glue release created:"
echo "${RELEASE_DIR}"
echo

find "${RELEASE_DIR}" -type f -printf '%P\n' | sort

echo
echo "Artifact bucket:"
echo "${ARTIFACT_BUCKET}"

echo
echo "S3 release prefix:"
echo "s3://${ARTIFACT_BUCKET}/glue/enterprise-ai-platform/${COMMIT_SHA}/"
