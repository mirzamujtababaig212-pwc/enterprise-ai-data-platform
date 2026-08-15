#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8001}"
API_KEY="${API_KEY:-super-secret-key}"

echo "========================================"
echo "Enterprise AI Gateway Smoke Test"
echo "========================================"
echo
echo "Base URL: ${BASE_URL}"
echo

echo "[1/8] Checking liveness..."

curl \
  --fail \
  --silent \
  --show-error \
  "${BASE_URL}/health"

echo
echo "PASS"
echo

echo "[2/8] Checking OpenAPI..."

curl \
  --fail \
  --silent \
  --show-error \
  "${BASE_URL}/openapi.json" \
  > /tmp/enterprise-ai-openapi.json

echo "PASS"
echo

echo "[3/8] Checking model registry..."
curl \
  --fail \
  --silent \
  --show-error \
  -H "x-api-key: ${API_KEY}" \
  "${BASE_URL}/v1/models" \
  | python -m json.tool
echo
echo "PASS"
echo

echo "[4/8] Checking authenticated health..."

curl \
  --fail \
  --silent \
  --show-error \
  -H "x-api-key: ${API_KEY}" \
  "${BASE_URL}/v1/health" \
  | python -m json.tool

echo
echo "PASS"
echo

echo "[5/8] Testing missing API key..."

STATUS_CODE="$(
  curl \
    --silent \
    --output /dev/null \
    --write-out "%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    "${BASE_URL}/v1/chat"
)"

if [[ "${STATUS_CODE}" != "401" ]]; then
  echo "FAIL: expected 401, received ${STATUS_CODE}"
  exit 1
fi

echo "PASS: received 401"
echo

echo "[6/8] Testing invalid API key..."

STATUS_CODE="$(
  curl \
    --silent \
    --output /dev/null \
    --write-out "%{http_code}" \
    -X POST \
    -H "Content-Type: application/json" \
    -H "x-api-key: invalid-key" \
    -d '{
      "provider": "mock",
      "model": "mock-gpt",
      "prompt": "Authentication test"
    }' \
    "${BASE_URL}/v1/chat"
)"

if [[ "${STATUS_CODE}" != "401" ]]; then
  echo "FAIL: expected 401, received ${STATUS_CODE}"
  exit 1
fi

echo "PASS: received 401"
echo

echo "[7/8] Testing authenticated mock chat..."

CHAT_RESPONSE="$(
  curl \
    --fail \
    --silent \
    --show-error \
    -X POST \
    -H "Content-Type: application/json" \
    -H "x-api-key: ${API_KEY}" \
    -d '{
      "provider": "mock",
      "model": "mock-gpt",
      "prompt": "Explain enterprise AI architecture in one sentence."
    }' \
    "${BASE_URL}/v1/chat"
)"

echo "${CHAT_RESPONSE}" | python -m json.tool

echo "${CHAT_RESPONSE}" | grep -q "Mock response"

echo "PASS"
echo

echo "[8/8] Checking Prometheus..."

curl \
  --fail \
  --silent \
  --show-error \
  "http://localhost:9090/-/healthy"

echo
echo "PASS"
echo

echo "========================================"
echo "ALL SMOKE TESTS PASSED"
echo "========================================"
