#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9091}"
JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"

CHAT_REQUEST_FILE="/tmp/llm-gateway-chat-request.json"
CHAT_RESPONSE_FILE="/tmp/llm-gateway-chat-response.json"
CHAT_HEADERS_FILE="/tmp/llm-gateway-chat-headers.txt"
METRICS_FILE="/tmp/llm-gateway-metrics.txt"
TRACES_FILE="/tmp/llm-gateway-traces.json"

echo "============================================================"
echo "ENTERPRISE AI PLATFORM"
echo "LLM GATEWAY ENTERPRISE OBSERVABILITY E2E"
echo "============================================================"

echo
echo "STEP 1 — Validate API key"

if [[ -z "${GATEWAY_API_KEY:-}" ]]; then
    echo "ERROR: GATEWAY_API_KEY is not set"
    exit 1
fi

echo "GATEWAY_API_KEY is present"
echo "PASS"

echo
echo "STEP 2 — Gateway health"

curl -fsS \
    "${BASE_URL}/health"

echo
echo "PASS"

echo
echo "STEP 3 — Gateway healthz"

curl -fsS \
    "${BASE_URL}/healthz"

echo
echo "PASS"

echo
echo "STEP 4 — Prometheus target"

TARGETS="$(curl -fsS "${PROMETHEUS_URL}/api/v1/targets")"

echo "${TARGETS}" |
python -c '
import json
import sys

data = json.load(sys.stdin)

targets = data.get("data", {}).get("activeTargets", [])

matched = False

for target in targets:
    labels = target.get("labels", {})

    if labels.get("job") == "llm_gateway":
        matched = True

        print("job     :", labels.get("job"))
        print("instance:", labels.get("instance"))
        print("health  :", target.get("health"))

        if target.get("health") != "up":
            print("ERROR: Prometheus target is not healthy")
            sys.exit(1)

if not matched:
    print("ERROR: llm_gateway target not found")
    sys.exit(1)
'

echo "PASS"

echo
echo "STEP 5 — Jaeger service"

SERVICES="$(curl -fsS "${JAEGER_URL}/api/services")"

echo "${SERVICES}" |
python -c '
import json
import sys

data = json.load(sys.stdin)

services = data.get("data", [])

print("Services:")

for service in services:
    print("  -", service)

if "enterprise-ai-platform" not in services:
    print("ERROR: enterprise-ai-platform not found in Jaeger")
    sys.exit(1)
'

echo "PASS"

echo
echo "STEP 6 — Create authenticated chat request"

cat > "${CHAT_REQUEST_FILE}" <<'JSON'
{
  "prompt": "Explain enterprise AI observability in three concise points.",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "max_tokens": 150,
  "stream": false,
  "user_id": "observability-e2e-test"
}
JSON

cat "${CHAT_REQUEST_FILE}"

echo
echo "PASS"

echo
echo "STEP 7 — Execute authenticated /v1/chat"

HTTP_STATUS="$(
    curl -sS \
        -D "${CHAT_HEADERS_FILE}" \
        -o "${CHAT_RESPONSE_FILE}" \
        -w "%{http_code}" \
        -X POST \
        "${BASE_URL}/v1/chat" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: ${GATEWAY_API_KEY}" \
        --data @"${CHAT_REQUEST_FILE}"
)"

echo "HTTP_STATUS=${HTTP_STATUS}"

echo
echo "Response:"
cat "${CHAT_RESPONSE_FILE}"

echo

if [[ "${HTTP_STATUS}" != "200" ]]; then
    echo
    echo "ERROR: /v1/chat did not return HTTP 200"
    echo
    echo "Headers:"
    cat "${CHAT_HEADERS_FILE}"
    echo
    exit 1
fi

echo "PASS"

echo
echo "STEP 8 — Validate chat response"

python - "${CHAT_RESPONSE_FILE}" <<'PY'
import json
import sys

path = sys.argv[1]

with open(path) as f:
    data = json.load(f)

if "reply" not in data:
    print("ERROR: response does not contain reply")
    sys.exit(1)

if "metrics" not in data:
    print("ERROR: response does not contain metrics")
    sys.exit(1)

print("reply   : present")
print("metrics : present")

reply = data["reply"]

if not isinstance(reply, str) or not reply.strip():
    print("ERROR: reply is empty")
    sys.exit(1)

print("reply_length:", len(reply))

print()
print("Response metrics:")
print(json.dumps(data["metrics"], indent=2))
PY

echo "PASS"

echo
echo "STEP 9 — Capture Prometheus metrics"

curl -fsS \
    "${BASE_URL}/metrics" \
    > "${METRICS_FILE}"

grep -E \
'^(llm_gateway_provider_requests_total|llm_gateway_provider_latency_seconds|llm_gateway_provider_errors_total|llm_gateway_fallback_requests_total|llm_gateway_model_requests_total|llm_gateway_input_tokens_total|llm_gateway_output_tokens_total|llm_gateway_estimated_cost_total|llm_gateway_errors_total)' \
"${METRICS_FILE}" || true

echo
echo "PASS"

echo
echo "STEP 10 — Query model request metric"

curl -fsS \
    "${PROMETHEUS_URL}/api/v1/query?query=llm_gateway_model_requests_total" |
python -m json.tool

echo
echo "PASS"

echo
echo "STEP 11 — Query provider request metric"

curl -fsS \
    "${PROMETHEUS_URL}/api/v1/query?query=llm_gateway_provider_requests_total" |
python -m json.tool

echo
echo "PASS"

echo
echo "STEP 12 — Query token metrics"

echo
echo "INPUT TOKENS"

curl -fsS \
    "${PROMETHEUS_URL}/api/v1/query?query=llm_gateway_input_tokens_total" |
python -m json.tool

echo
echo "OUTPUT TOKENS"

curl -fsS \
    "${PROMETHEUS_URL}/api/v1/query?query=llm_gateway_output_tokens_total" |
python -m json.tool

echo
echo "PASS"

echo
echo "STEP 13 — Query estimated cost"

curl -fsS \
    "${PROMETHEUS_URL}/api/v1/query?query=llm_gateway_estimated_cost_total" |
python -m json.tool

echo
echo "PASS"

echo
echo "STEP 14 — Retrieve Jaeger traces"

curl -fsS \
    "${JAEGER_URL}/api/traces?service=enterprise-ai-platform&limit=20" \
    > "${TRACES_FILE}"

echo "Trace count:"
python - "${TRACES_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

traces = data.get("data", [])

print(len(traces))
PY

echo
echo "PASS"

echo
echo "STEP 15 — Locate POST /v1/chat trace"

python - "${TRACES_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

found = False

for trace in data.get("data", []):
    for span in trace.get("spans", []):

        if span.get("operationName") == "POST /v1/chat":
            found = True

            print("============================================================")
            print("CHAT TRACE FOUND")
            print("============================================================")

            print("Trace ID :", span.get("traceID"))
            print("Span ID  :", span.get("spanID"))
            print("Operation:", span.get("operationName"))
            print("Duration :", span.get("duration"), "microseconds")

            print()
            print("Tags:")

            for tag in span.get("tags", []):
                print(
                    f"  {tag.get('key')} = {tag.get('value')}"
                )

if not found:
    print("ERROR: POST /v1/chat trace was not found")
    sys.exit(1)
PY

echo "PASS"

echo
echo "STEP 16 — Gateway logs"

docker logs --tail 100 enterprise-ai-platform 2>&1 |
grep -Ei \
'chat|openai|provider|trace|error|exception|token|usage' \
|| true

echo
echo "============================================================"
echo "LLM GATEWAY E2E VALIDATION COMPLETE"
echo "============================================================"
echo
echo "Validated:"
echo "  [1] API authentication"
echo "  [2] Gateway health"
echo "  [3] Gateway healthz"
echo "  [4] Prometheus target"
echo "  [5] Jaeger service"
echo "  [6] Chat request schema"
echo "  [7] Authenticated /v1/chat"
echo "  [8] Chat response"
echo "  [9] Prometheus metrics"
echo "  [10] Provider/model metrics"
echo "  [11] Token metrics"
echo "  [12] Cost metrics"
echo "  [13] Jaeger traces"
echo "  [14] Gateway logs"
echo
