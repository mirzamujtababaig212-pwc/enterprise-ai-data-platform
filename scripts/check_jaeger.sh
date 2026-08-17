#!/usr/bin/env bash

set -u

JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"

echo
echo "============================================================"
echo " Jaeger Inspection"
echo "============================================================"
echo

echo "[1] Jaeger health"
echo "------------------------------------------------------------"

curl -fsS \
  "${JAEGER_URL}/" \
  -o /dev/null

if [ "$?" -eq 0 ]; then
    echo "Jaeger UI/API reachable."
else
    echo "ERROR: Jaeger is not reachable."
    exit 1
fi

echo
echo "[2] Services"
echo "------------------------------------------------------------"

curl -fsS \
  "${JAEGER_URL}/api/services" \
  | python -m json.tool

echo
echo "[3] Raw services JSON"
echo "------------------------------------------------------------"

curl -fsS \
  "${JAEGER_URL}/api/services"

echo
echo
echo "============================================================"
echo " Jaeger inspection complete"
echo "============================================================"
echo
