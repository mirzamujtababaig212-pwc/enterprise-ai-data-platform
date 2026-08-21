#!/usr/bin/env bash

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://localhost:9091}"
JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"

echo "============================================================"
echo "ENTERPRISE AI PLATFORM"
echo "LLM GATEWAY END-TO-END VALIDATION"
echo "============================================================"

echo
echo "STEP 1 — Gateway health"
curl -fsS "${BASE_URL}/health"
echo
echo "PASS"

echo
echo "STEP 2 — Gateway healthz"
curl -fsS "${BASE_URL}/healthz"
echo
echo "PASS"

echo
echo "STEP 3 — Prometheus target"
TARGETS="$(curl -fsS "${PROMETHEUS_URL}/api/v1/targets")"

echo "${TARGETS}" | \
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
        print("job:", labels.get("job"))
        print("instance:", labels.get("instance"))
        print("health:", target.get("health"))

if not matched:
    print("ERROR: llm_gateway target not found")
    sys.exit(1)
'

echo "PASS"

echo
echo "STEP 4 — Jaeger services"
SERVICES="$(curl -fsS "${JAEGER_URL}/api/services")"

echo "${SERVICES}" | \
python -c '
import json
import sys

data = json.load(sys.stdin)
services = data.get("data", [])

print("Services:")
for service in services:
    print("  -", service)
'

echo
echo "============================================================"
echo "INFRASTRUCTURE VALIDATION PASSED"
echo "============================================================"
echo
echo "Next:"
echo "  1. Set GATEWAY_API_KEY"
echo "  2. Execute authenticated /v1/chat"
echo "  3. Verify real provider response"
echo "  4. Verify Prometheus metrics"
echo "  5. Verify OTel trace"
echo "  6. Verify Jaeger trace"
echo
