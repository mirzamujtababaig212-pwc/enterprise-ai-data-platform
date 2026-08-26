#!/usr/bin/env bash

set -euo pipefail

PROJECT_NETWORK="enterprise_ai_platform_default"
APP_CONTAINER="enterprise-ai-platform"
JAEGER_HOST="jaeger"
JAEGER_PORT="16686"
SERVICE_NAME="enterprise-ai-platform"

TMP_RESPONSE="/tmp/otel-runtime-response.json"
TMP_TRACES="/tmp/otel-jaeger-traces.json"
TMP_EXACT="/tmp/otel-exact-trace.json"


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
        -o "$TMP_RESPONSE" \
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
    cat "$TMP_RESPONSE"
    exit 1
fi

echo "PASS: Proof request returned HTTP 200"

sleep 7

echo
echo "=== STEP 5: Query recent Jaeger traces ==="

docker run --rm \
  --network "$PROJECT_NETWORK" \
  curlimages/curl:latest \
  -sS \
  "http://${JAEGER_HOST}:${JAEGER_PORT}/api/traces?service=${SERVICE_NAME}&lookback=2m&limit=100" \
  > "$TMP_TRACES"

echo
echo "=== STEP 6: Identify newest application trace ==="

TRACE_ID="$(
python - "$TRACE_PROOF_ID" "$TMP_TRACES" <<'PY'
import json
import sys

proof_id = sys.argv[1]
path = sys.argv[2]

with open(path) as f:
    data = json.load(f)

matches = []

for trace in data.get("data", []):
    trace_id = trace.get("traceID")

    for span in trace.get("spans", []):
        if span.get("operationName") != "POST /v1/test-body":
            continue

        for tag in span.get("tags", []):
            if (
                tag.get("key") == "validation.proof_id"
                and tag.get("value") == proof_id
            ):
                matches.append(trace_id)

        for log in span.get("logs", []):
            for field in log.get("fields", []):
                if (
                    field.get("key") == "validation.proof_id"
                    and field.get("value") == proof_id
                ):
                    matches.append(trace_id)

if not matches:
    print(
        "FAIL: No Jaeger trace correlated with proof ID",
        file=sys.stderr,
    )
    print(f"Proof ID: {proof_id}", file=sys.stderr)
    sys.exit(1)

matches = list(dict.fromkeys(matches))

if len(matches) != 1:
    print(
        f"FAIL: Expected exactly one matching trace, found {len(matches)}",
        file=sys.stderr,
    )
    for trace_id in matches:
        print(trace_id, file=sys.stderr)
    sys.exit(1)

print(matches[0])
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
    > "$TMP_EXACT"

echo
echo "=== STEP 8: Validate exact trace ==="

python - "$TRACE_ID" "$TRACE_PROOF_ID" "$TMP_EXACT" <<'PY'
import json
import sys

expected_trace_id = sys.argv[1]
expected_proof_id = sys.argv[2]
path = sys.argv[3]

with open(path) as f:
    data = json.load(f)

traces = data.get("data", [])

if not traces:
    print("FAIL: Exact trace not found in Jaeger")
    sys.exit(1)

trace = traces[0]

if trace.get("traceID") != expected_trace_id:
    print("FAIL: Wrong trace returned")
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

for span in spans:
    print("SPAN:", span.get("operationName"))

print()
print("=== ASSERTIONS ===")

if "POST /v1/test-body" not in operations:
    print("FAIL: Application span missing")
    sys.exit(1)

application_spans = [
    span
    for span in spans
    if span.get("operationName") == "POST /v1/test-body"
]

if len(application_spans) != 1:
    print(
        "FAIL: Expected exactly one application span, "
        f"found {len(application_spans)}"
    )
    sys.exit(1)

application_span = application_spans[0]

proof_values = [
    tag.get("value")
    for tag in application_span.get("tags", [])
    if tag.get("key") == "validation.proof_id"
]

if expected_proof_id not in proof_values:
    print("FAIL: Proof ID missing from application span")
    print("Expected:", expected_proof_id)
    print("Found:", proof_values)
    sys.exit(1)

authentication_spans = [
    span
    for span in spans
    if span.get("operationName") == "authentication"
]

if len(authentication_spans) != 1:
    print(
        "FAIL: Expected exactly one authentication span, "
        f"found {len(authentication_spans)}"
    )
    sys.exit(1)

authentication_span = authentication_spans[0]

references = authentication_span.get("references", [])

if not any(
    ref.get("refType") == "CHILD_OF"
    and ref.get("spanID") == application_span.get("spanID")
    for ref in references
):
    print(
        "FAIL: authentication span is not a child of "
        "the application span"
    )
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
print("PASS: Proof ID correlated to exact trace")
print("PASS: POST /v1/test-body exported")
print("PASS: authentication span exported")
print("PASS: authentication is child of application span")
print("PASS: http receive noise absent")
print("PASS: http send noise absent")
print("PASS: GET /metrics absent")
PY

echo
echo "=== STEP 9: Collector evidence ==="

COLLECTOR_OUTPUT="$(
    docker logs otel-collector --since 5m 2>&1 \
        | grep -F '"otelcol.signal": "traces"' \
        | tail -10 || true
)"

if [[ -n "$COLLECTOR_OUTPUT" ]]; then
    echo "$COLLECTOR_OUTPUT"
    echo "PASS: Collector emitted trace telemetry"
else
    echo "WARNING: No matching collector debug trace log found"
fi

echo
echo "============================================================"
echo "OTEL END-TO-END VALIDATION COMPLETE"
echo "============================================================"

echo "TRACE_PROOF_ID=${TRACE_PROOF_ID}"
echo "TRACE_ID=${TRACE_ID}"
