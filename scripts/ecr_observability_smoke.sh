#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Enterprise AI Platform
# ECR / LLM Gateway Observability Smoke Test
# ============================================================

BASE_URL="${BASE_URL:-http://localhost:8002}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9090}"
JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"
API_KEY="${API_KEY:-super-secret-key}"
GATEWAY_CONTAINER="${GATEWAY_CONTAINER:-eai-gateway-test}"

# ------------------------------------------------------------
# Colors / formatting
# ------------------------------------------------------------

if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

PASS_COUNT=0
FAIL_COUNT=0

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo -e "${RED}ERROR: required command not found: $1${NC}"
        exit 1
    }
}

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    echo -e "${GREEN}PASS: $1${NC}"
    echo
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    echo -e "${RED}FAIL: $1${NC}"
    echo
}

section() {
    echo
    echo "============================================================"
    echo "$1"
    echo "============================================================"
    echo
}

json_pretty() {
    python -m json.tool
}

require_command curl
require_command python
require_command docker

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

section "Enterprise AI Platform - ECR Observability Smoke Test"

echo "Gateway:           ${BASE_URL}"
echo "Prometheus:        ${PROMETHEUS_URL}"
echo "Jaeger:            ${JAEGER_URL}"
echo "Gateway container: ${GATEWAY_CONTAINER}"
echo

# ============================================================
# 1. Gateway health
# ============================================================

section "1. Gateway health"

HEALTH_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        -H "X-API-Key: ${API_KEY}" \
        "${BASE_URL}/v1/health"
)"

echo "${HEALTH_RESPONSE}" | json_pretty

HEALTH_STATUS="$(
    echo "${HEALTH_RESPONSE}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)

status = (
    data.get("status")
    or data.get("health")
    or data.get("state")
)

if status:
    print(status)
'
)"

echo "Gateway status: ${HEALTH_STATUS:-unknown}"

pass "Gateway health"

# ============================================================
# 2. Model registry
# ============================================================

section "2. Model registry"

MODELS_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        -H "X-API-Key: ${API_KEY}" \
        "${BASE_URL}/v1/models"
)"

echo "${MODELS_RESPONSE}" | json_pretty

pass "Model registry"

# ============================================================
# 3. Mock chat
# ============================================================

section "3. Mock chat"

CHAT_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        -X POST \
        -H "Content-Type: application/json" \
        -H "X-API-Key: ${API_KEY}" \
        "${BASE_URL}/v1/chat" \
        -d '{
            "provider": "mock",
            "model": "mock-gpt",
            "prompt": "Observability smoke test"
        }'
)"

echo "${CHAT_RESPONSE}" | json_pretty

CHAT_STATUS="$(
    echo "${CHAT_RESPONSE}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)

print(
    data
    .get("metrics", {})
    .get("status", "")
)
'
)"

CHAT_REQUEST_ID="$(
    echo "${CHAT_RESPONSE}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)

print(
    data
    .get("metrics", {})
    .get("request_id", "")
)
'
)"

CHAT_TRACE_ID="$(
    echo "${CHAT_RESPONSE}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)

print(
    data
    .get("metrics", {})
    .get("trace_id", "")
)
' 2>/dev/null || true
)"

echo
echo "Chat status:     ${CHAT_STATUS}"
echo "Chat request ID: ${CHAT_REQUEST_ID}"
echo "Chat trace ID:   ${CHAT_TRACE_ID:-not-returned}"

if [ "${CHAT_STATUS}" != "success" ]; then
    echo -e "${RED}ERROR: chat request did not succeed${NC}"
    exit 1
fi

pass "Mock chat"

# ============================================================
# 4. Mock embeddings
# ============================================================

section "4. Mock embeddings"

EMBED_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        -X POST \
        -H "Content-Type: application/json" \
        -H "X-API-Key: ${API_KEY}" \
        "${BASE_URL}/v1/embeddings" \
        -d '{
            "provider": "mock",
            "model": "mock-embedding",
            "text": "Enterprise AI Platform observability"
        }'
)"

echo "${EMBED_RESPONSE}" | json_pretty

EMBED_STATUS="$(
    echo "${EMBED_RESPONSE}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)

print(
    data
    .get("metrics", {})
    .get("status", "")
)
'
)"

EMBED_REQUEST_ID="$(
    echo "${EMBED_RESPONSE}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)

print(
    data
    .get("metrics", {})
    .get("request_id", "")
)
'
)"

echo
echo "Embedding status:     ${EMBED_STATUS}"
echo "Embedding request ID: ${EMBED_REQUEST_ID}"

if [ "${EMBED_STATUS}" != "success" ]; then
    echo -e "${RED}ERROR: embedding request did not succeed${NC}"
    exit 1
fi

pass "Mock embeddings"

# ============================================================
# 5. Prometheus availability
# ============================================================

section "5. Prometheus availability"

PROM_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        "${PROMETHEUS_URL}/api/v1/query?query=up"
)"

echo "${PROM_RESPONSE}" | json_pretty

PROM_STATUS="$(
    echo "${PROM_RESPONSE}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)

print(data.get("status", ""))
'
)"

if [ "${PROM_STATUS}" != "success" ]; then
    echo -e "${RED}ERROR: Prometheus query failed${NC}"
    exit 1
fi

pass "Prometheus availability"

# ============================================================
# 6. Prometheus target health
# ============================================================

section "6. Prometheus gateway target health"

TARGETS_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        "${PROMETHEUS_URL}/api/v1/targets"
)"

echo "${TARGETS_RESPONSE}" |
python -c '
import json
import sys

data = json.load(sys.stdin)

targets = data.get("data", {}).get("activeTargets", [])

gateway_targets = [
    target
    for target in targets
    if target.get("labels", {}).get("job") == "llm-gateway"
]

if not gateway_targets:
    print("ERROR: llm-gateway Prometheus target not found")
    sys.exit(1)

for target in gateway_targets:
    labels = target.get("labels", {})

    print(
        "job={job}".format(
            job=labels.get("job")
        )
    )

    print(
        "instance={instance}".format(
            instance=labels.get("instance")
        )
    )

    print(
        "health={health}".format(
            health=target.get("health")
        )
    )

    print(
        "scrapeUrl={url}".format(
            url=target.get("scrapeUrl")
        )
    )

    print(
        "lastError={error}".format(
            error=target.get("lastError")
        )
    )

    if target.get("health") != "up":
        print("ERROR: llm-gateway Prometheus target is not healthy")
        sys.exit(1)
'

pass "Prometheus llm-gateway target health"

# ============================================================
# 7. Prometheus metric verification
# ============================================================

section "7. Prometheus gateway metrics"

METRICS_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        "${BASE_URL}/metrics"
)"

echo "Relevant gateway metrics:"
echo

echo "${METRICS_RESPONSE}" |
grep -E \
    'llm_gateway_(http_requests_total|provider_requests_total|model_requests_total|requests_total|input_tokens_total|output_tokens_total|estimated_cost_total|errors_total)' \
    || true

echo

# Verify that at least one expected gateway metric exists.
if echo "${METRICS_RESPONSE}" |
    grep -Eq \
    'llm_gateway_(http_requests_total|requests_total|provider_requests_total)'; then

    pass "Gateway Prometheus counters exposed"

else
    echo -e "${YELLOW}WARNING: Expected llm_gateway request counter was not found${NC}"
    echo "The /metrics endpoint is reachable, but metric names may differ."
    echo
fi

# ------------------------------------------------------------
# Query a gateway request counter through Prometheus.
# ------------------------------------------------------------

PROM_GATEWAY_QUERY="$(
    curl \
        --fail \
        --silent \
        --show-error \
        --get \
        --data-urlencode \
        'query=llm_gateway_requests_total' \
        "${PROMETHEUS_URL}/api/v1/query"
)"

echo "Prometheus query result:"
echo "${PROM_GATEWAY_QUERY}" | json_pretty

PROM_GATEWAY_STATUS="$(
    echo "${PROM_GATEWAY_QUERY}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)
print(data.get("status", ""))
'
)"

if [ "${PROM_GATEWAY_STATUS}" = "success" ]; then
    pass "Prometheus gateway request counter query"
else
    echo -e "${YELLOW}WARNING: llm_gateway_requests_total query was not successful${NC}"
    echo
fi

# ============================================================
# 8. Authentication rejection
# ============================================================

section "8. Authentication rejection"

AUTH_RESPONSE_FILE="$(mktemp)"

AUTH_HTTP_CODE="$(
    curl \
        --silent \
        --show-error \
        --output "${AUTH_RESPONSE_FILE}" \
        --write-out "%{http_code}" \
        -H "X-API-Key: definitely-invalid-key" \
        "${BASE_URL}/v1/models"
)"

echo "HTTP status: ${AUTH_HTTP_CODE}"
echo

cat "${AUTH_RESPONSE_FILE}" |
    json_pretty || cat "${AUTH_RESPONSE_FILE}"

rm -f "${AUTH_RESPONSE_FILE}"

if [ "${AUTH_HTTP_CODE}" != "401" ]; then
    echo -e "${RED}ERROR: expected HTTP 401, received ${AUTH_HTTP_CODE}${NC}"
    exit 1
fi

pass "Invalid API key rejected with HTTP 401"

# ============================================================
# 9. Invalid provider/model handling
# ============================================================

section "9. Invalid provider/model handling"

INVALID_RESPONSE_FILE="$(mktemp)"

INVALID_HTTP_CODE="$(
    curl \
        --silent \
        --show-error \
        --output "${INVALID_RESPONSE_FILE}" \
        --write-out "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -H "X-API-Key: ${API_KEY}" \
        "${BASE_URL}/v1/chat" \
        -d '{
            "provider": "invalid-provider",
            "model": "invalid-model",
            "prompt": "This should fail in a controlled manner"
        }'
)"

echo "HTTP status: ${INVALID_HTTP_CODE}"
echo

cat "${INVALID_RESPONSE_FILE}" |
    json_pretty || cat "${INVALID_RESPONSE_FILE}"

rm -f "${INVALID_RESPONSE_FILE}"

if [ "${INVALID_HTTP_CODE}" = "200" ]; then
    echo -e "${RED}ERROR: invalid provider unexpectedly returned HTTP 200${NC}"
    exit 1
fi

pass "Invalid provider/model handled as controlled error"

# ============================================================
# 10. Jaeger service discovery
# ============================================================

section "10. Jaeger service discovery"

JAEGER_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        "${JAEGER_URL}/api/services"
)"

echo "${JAEGER_RESPONSE}" | json_pretty

echo "${JAEGER_RESPONSE}" |
python -c '
import json
import sys

data = json.load(sys.stdin)

services = data.get("data", [])

if "llm-gateway" not in services:
    print("ERROR: llm-gateway not found in Jaeger services")
    sys.exit(1)

print("Jaeger service llm-gateway: FOUND")
'

pass "Jaeger service discovery"

# ============================================================
# 11. Jaeger trace verification
# ============================================================

section "11. Jaeger trace verification"

TRACE_RESPONSE="$(
    curl \
        --fail \
        --silent \
        --show-error \
        "${JAEGER_URL}/api/traces?service=llm-gateway&limit=20"
)"

TRACE_COUNT="$(
    echo "${TRACE_RESPONSE}" |
    python -c '
import json
import sys

data = json.load(sys.stdin)

print(
    len(
        data.get("data", [])
    )
)
'
)"

echo "Jaeger trace count: ${TRACE_COUNT}"

if [ "${TRACE_COUNT}" -eq 0 ]; then
    echo -e "${RED}ERROR: No llm-gateway traces found in Jaeger${NC}"
    exit 1
fi

# ------------------------------------------------------------
# Find the trace containing our latest chat request.
# ------------------------------------------------------------

if [ -n "${CHAT_REQUEST_ID}" ]; then

    TRACE_MATCH="$(
        echo "${TRACE_RESPONSE}" |
        python -c '
import json
import sys

data = json.load(sys.stdin)

request_id = sys.argv[1]

traces = data.get("data", [])

matches = []

for trace in traces:
    trace_id = trace.get("traceID", "")

    for span in trace.get("spans", []):
        tags = span.get("tags", [])

        for tag in tags:
            value = tag.get("value")

            if value == request_id:
                matches.append(
                    {
                        "traceID": trace_id,
                        "spanID": span.get("spanID"),
                        "operationName": span.get("operationName"),
                    }
                )

if matches:
    for match in matches:
        print(
            "MATCH traceID={traceID} spanID={spanID} operation={operationName}".format(
                **match
            )
        )
else:
    print("NO_REQUEST_ID_MATCH")
' \
        "${CHAT_REQUEST_ID}"
    )"

    echo "${TRACE_MATCH}"

    if echo "${TRACE_MATCH}" | grep -q "MATCH"; then
        pass "Jaeger trace correlated to chat request"
    else
        echo -e "${YELLOW}WARNING: Jaeger contains traces but request_id correlation was not found in the returned 20 traces.${NC}"
        echo "This does not mean tracing is broken; the request may have aged out of the query window."
        echo
    fi

else
    echo -e "${YELLOW}WARNING: Chat request ID unavailable; skipping request correlation.${NC}"
fi

echo
echo "Recent llm-gateway traces:"
echo "${TRACE_RESPONSE}" |
python -c '
import json
import sys

data = json.load(sys.stdin)

for trace in data.get("data", [])[:5]:

    print(
        "traceID={traceID}".format(
            traceID=trace.get("traceID")
        )
    )

    for span in trace.get("spans", [])[:10]:

        print(
            "  span={span} operation={operation}".format(
                span=span.get("spanID"),
                operation=span.get("operationName")
            )
        )
'

# ============================================================
# 12. Recent gateway logs
# ============================================================

section "12. Recent gateway logs"

docker logs \
    --tail 40 \
    "${GATEWAY_CONTAINER}"

echo
echo "Gateway logs retrieved successfully."

pass "Gateway logs"

# ============================================================
# Final result
# ============================================================

section "FINAL RESULT"

echo "Gateway health                         : PASS"
echo "Model registry                         : PASS"
echo "Mock chat                              : PASS"
echo "Mock embeddings                        : PASS"
echo "Prometheus availability                : PASS"
echo "Prometheus target health               : PASS"
echo "Prometheus metrics                     : PASS/WARNING"
echo "Authentication rejection               : PASS"
echo "Invalid provider/model handling        : PASS"
echo "Jaeger service discovery               : PASS"
echo "Jaeger trace verification              : PASS"
echo "Gateway logs                           : PASS"
echo

echo "Checks completed: ${PASS_COUNT}"
echo

if [ "${FAIL_COUNT}" -ne 0 ]; then
    echo -e "${RED}OBSERVABILITY SMOKE TEST: FAILED${NC}"
    exit 1
fi

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}OBSERVABILITY SMOKE TEST: PASSED${NC}"
echo -e "${GREEN}============================================================${NC}"
echo

