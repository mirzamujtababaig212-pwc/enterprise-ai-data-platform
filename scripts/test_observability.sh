#!/usr/bin/env bash

set -u

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8002}"
JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"
API_KEY="${API_KEY:-super-secret-key}"

echo
echo "============================================================"
echo " Enterprise AI Platform - Observability Smoke Test"
echo "============================================================"
echo

echo "[1/7] Checking gateway health..."
echo

curl -fsS \
  -H "X-API-Key: ${API_KEY}" \
  "${GATEWAY_URL}/v1/health" \
  | python -m json.tool

echo
echo "------------------------------------------------------------"
echo

echo "[2/7] Checking available models..."
echo

curl -fsS \
  -H "X-API-Key: ${API_KEY}" \
  "${GATEWAY_URL}/v1/models" \
  | python -m json.tool

echo
echo "------------------------------------------------------------"
echo

echo "[3/7] Sending mock chat request..."
echo

CHAT_RESPONSE="$(
  curl -fsS \
    -X POST \
    -H "Content-Type: application/json" \
    -H "X-API-Key: ${API_KEY}" \
    "${GATEWAY_URL}/v1/chat" \
    -d '{
      "provider": "mock",
      "model": "mock-gpt",
      "prompt": "Observability smoke test"
    }'
)"

echo "${CHAT_RESPONSE}" | python -m json.tool

REQUEST_ID="$(
  echo "${CHAT_RESPONSE}" |
  python -c '
import json
import sys

data = json.load(sys.stdin)

print(
    data.get("metrics", {}).get(
        "request_id",
        data.get("request_id", "")
    )
)
'
)"

echo
echo "Chat request ID:"
echo "${REQUEST_ID}"

echo
echo "------------------------------------------------------------"
echo

echo "[4/7] Sending mock embedding request..."
echo

curl -fsS \
  -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${API_KEY}" \
  "${GATEWAY_URL}/v1/embeddings" \
  -d '{
    "provider": "mock",
    "model": "mock-embedding",
    "text": "Observability smoke test"
  }' | python -m json.tool

echo
echo "------------------------------------------------------------"
echo

echo "[5/7] Checking Jaeger services..."
echo

SERVICES_RESPONSE="$(
  curl -fsS \
    "${JAEGER_URL}/api/services"
)"

echo "${SERVICES_RESPONSE}" | python -m json.tool

echo
echo "------------------------------------------------------------"
echo

echo "[6/7] Recent gateway logs..."
echo

docker logs --tail 50 eai-gateway-test 2>&1

echo
echo "------------------------------------------------------------"
echo

echo "[7/7] Test summary"
echo

if [ -n "${REQUEST_ID}" ]; then
    echo "SUCCESS: Gateway request completed."
    echo "Request ID: ${REQUEST_ID}"
else
    echo "WARNING: Could not extract request ID."
fi

echo
echo "Jaeger UI:"
echo "${JAEGER_URL}"
echo

echo "============================================================"
