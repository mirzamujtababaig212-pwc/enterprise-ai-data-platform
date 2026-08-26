#!/usr/bin/env bash

set -euo pipefail

AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="$(aws sts get-caller-identity \
  --query Account \
  --output text)"

ECR_REPOSITORY="${ECR_REPOSITORY:-enterprise-ai-platform}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"

ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

echo "AWS account: ${AWS_ACCOUNT_ID}"
echo "AWS region:  ${AWS_REGION}"
echo "ECR URI:     ${ECR_URI}"
echo "Image tag:   ${IMAGE_TAG}"

aws ecr describe-repositories \
  --repository-names "${ECR_REPOSITORY}" \
  --region "${AWS_REGION}" \
  >/dev/null 2>&1 || \
aws ecr create-repository \
  --repository-name "${ECR_REPOSITORY}" \
  --region "${AWS_REGION}" \
  >/dev/null

aws ecr get-login-password \
  --region "${AWS_REGION}" |
docker login \
  --username AWS \
  --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

docker build \
  --tag "${ECR_URI}:${IMAGE_TAG}" \
  .

docker push "${ECR_URI}:${IMAGE_TAG}"

echo
echo "Image pushed:"
echo "${ECR_URI}:${IMAGE_TAG}"
