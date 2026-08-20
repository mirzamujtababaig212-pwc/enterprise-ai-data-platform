#!/usr/bin/env bash

set -euo pipefail

PROJECT_NETWORK="enterprise_ai_platform_default"
APP_CONTAINER="enterprise-ai-platform"
JAEGER_HOST="jaeger"
JAEGER_PORT="16686"

echo "============================================================"
echo "OTEL END-TO-END RUNTIME VALIDATION"
echo "============================================================"

echo
echo "=== STEP 1: Validate containers ==="

docker compose ps \
  enterprise-ai-platform \
  otel-collector \
  jaeger

echo
echo "=== STEP 2: Validate OTEL environment ==="

docker exec "$APP_CONTAINER" printenv \
  | grep -E '^OTEL_|^SERVICE' \
  | sort

echo
echo "=== STEP 3: Validate collector connectivity ==="

docker exec "$APP_CONTAINER" python - <<'PY'
import socket

host = "otel-collector"
port = 4317

sock = socket.create_connection((host, port), timeout=5)
sock.close()

print(f"PASS: {host}:{port} reachable")
PY

echo
echo "=== STEP 4: Generate unique proof request ==="

TRACE_PROOF_ID="otel-automated-proof-$(date +%s)-$(openssl rand -hex 8)"

API_KEY_VALUE="$(
  docker exec "$APP_CONTAINER" printenv API_KEY
)"

echo "TRACE_PROOF_ID=${TRACE_PROOF_ID}"

HTTP_STATUS="$(
  curl -sS \
    -o /tmp/otel-runtime-response.json \
    -w '%{http_code}' \
    -X POST \
    http://127.0.0.1:8000/v1/test-body \
    -H "x-api-key: ${API_KEY_VALUE}" \
    -H "Content-Type: application/json" \
    -H "X-OTEL-Proof-ID: ${TRACE_PROOF_ID}" \
    -d "{\"test\":\"${TRACE_PROOF_ID}\"}"
)"

echo "HTTP_STATUS=${HTTP_STATUS}"

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "FAIL: Expected HTTP 200"
    cat /tmp/otel-runtime-response.json
    exit 1
fi

echo "PASS: Proof request returned HTTP 200"

echo
echo "=== STEP 5: Query recent Jaeger traces ==="

docker run --rm \
  --network "$PROJECT_NETWORK" \
  curlimages/curl:latest \
  -sS \
  "http://${JAEGER_HOST}:${JAEGER_PORT}/api/traces?service=enterprise-ai-platform&lookback=5m&limit=50" \
  > /tmp/otel-jaeger-traces.json

echo
echo "=== STEP 6: Identify newest application trace ==="

TRACE_ID="$(
python - <<'PY'
import json
import sys

with open("/tmp/otel-jaeger-traces.json") as f:
    data = json.load(f)

candidates = []

for trace in data.get("data", []):
    spans = trace.get("spans", [])

    for span in spans:
        if span.get("operationName") == "POST /v1/test-body":
            candidates.append(
                (
                    span.get("startTime", 0),
                    trace.get("traceID"),
                )
            )
            break

if not candidates:
    print("FAIL: No POST /v1/test-body trace found", file=sys.stderr)
    sys.exit(1)

candidates.sort(reverse=True)

print(candidates[0][1])
PY
)"

echo "TRACE_ID=${TRACE_ID}"

if [[ -z "$TRACE_ID" ]]; then
    echo "FAIL: TRACE_ID is empty"
    exit 1
fi

echo
echo "=== STEP 7: Retrieve exact trace ==="

docker run --rm \
  --network "$PROJECT_NETWORK" \
  curlimages/curl:latest \
  -sS \
  "http://${JAEGER_HOST}:${JAEGER_PORT}/api/traces/${TRACE_ID}" \
  > /tmp/otel-exact-trace.json

echo
echo "=== STEP 8: Validate exported trace ==="

python - <<PY
import json
import sys

expected_trace_id = "${TRACE_ID}"

with open("/tmp/otel-exact-trace.json") as f:
    data = json.load(f)

traces = data.get("data", [])

if not traces:
    print("FAIL: Exact trace not found in Jaeger")
    sys.exit(1)

trace = traces[0]

if trace.get("traceID") != expected_trace_id:
    print("FAIL: Jaeger returned unexpected trace ID")
    print("Expected:", expected_trace_id)
    print("Actual:", trace.get("traceID"))
    sys.exit(1)

spans = trace.get("spans", [])
operations = [
    span.get("operationName", "")
    for span in spans
]

print("Trace ID:", trace.get("traceID"))
print("Span count:", len(spans))
print()

for operation in operations:
    print("SPAN:", operation)

print()
print("=== ASSERTIONS ===")

if "POST /v1/test-body" not in operations:
    print("FAIL: Application span missing")
    sys.exit(1)

noise = [
    operation
    for operation in operations
    if operation.endswith(" http receive")
    or operation.endswith(" http send")
    or operation == "GET /metrics"
]

if noise:
    print("FAIL: Noise spans detected:")
    for operation in noise:
        print("  ", operation)
    sys.exit(1)

print("PASS: Exact trace found")
print("PASS: POST /v1/test-body exported")
print("PASS: http receive noise absent")
print("PASS: http send noise absent")
print("PASS: GET /metrics absent")
PY

echo
echo "=== STEP 9: Collector evidence ==="

docker logs otel-collector --since 5m 2>&1 \
  | grep -F '"otelcol.signal": "traces"' \
  | tail -10 || true

echo
echo "============================================================"
echo "OTEL END-TO-END VALIDATION PASSED"
echo "============================================================"

echo "TRACE_ID=${TRACE_ID}"
