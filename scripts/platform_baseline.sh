#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "============================================================"
echo " Enterprise AI Platform - Baseline Audit"
echo "============================================================"
echo

echo "Repository:"
echo "  $ROOT_DIR"
echo

echo "Git:"
git branch --show-current
git log -1 --oneline
git status --short
echo

echo "------------------------------------------------------------"
echo "1. Repository structure"
echo "------------------------------------------------------------"

for path in \
    Dockerfile \
    docker-compose.yml \
    Makefile \
    README.md \
    entrypoint.sh \
    requirements \
    config \
    common \
    ai_platform \
    tests \
    docs \
    monitoring \
    .github/workflows
do
    if [[ -e "$path" ]]; then
        echo "[OK]   $path"
    else
        echo "[MISS] $path"
    fi
done

echo

echo "------------------------------------------------------------"
echo "2. Docker"
echo "------------------------------------------------------------"

if command -v docker >/dev/null 2>&1; then
    docker --version
    docker compose version

    echo
    echo "Compose services:"
    docker compose config --services || true

    echo
    echo "Compose validation:"
    docker compose config >/tmp/enterprise_ai_platform_compose_check.yml

    if [[ $? -eq 0 ]]; then
        echo "[OK] docker compose config"
    else
        echo "[FAIL] docker compose config"
    fi
else
    echo "[WARN] Docker not available"
fi

echo

echo "------------------------------------------------------------"
echo "3. Python tooling"
echo "------------------------------------------------------------"

for command_name in python pytest ruff black mypy; do
    if command -v "$command_name" >/dev/null 2>&1; then
        echo
        echo "[$command_name]"
        "$command_name" --version || true
    else
        echo "[MISS] $command_name"
    fi
done

echo

echo "------------------------------------------------------------"
echo "4. Configuration files"
echo "------------------------------------------------------------"

find config -maxdepth 3 -type f 2>/dev/null | sort || true

echo

echo "Potential hardcoded environment-sensitive values:"
echo

grep -RInE \
    --exclude-dir=.git \
    --exclude-dir=.venv \
    --exclude-dir=__pycache__ \
    --exclude='*.pyc' \
    --exclude='platform_baseline.sh' \
    'localhost|127\.0\.0\.1|:9092|postgres|jdbc:|s3://|checkpoint' \
    common ai_platform config 2>/dev/null || true

echo

echo "------------------------------------------------------------"
echo "5. Observability"
echo "------------------------------------------------------------"

for path in \
    common/metrics \
    common/monitoring \
    ai_platform/llm_gateway/observability \
    monitoring/prometheus.yml \
    monitoring/alerts.yml
do
    if [[ -e "$path" ]]; then
        echo "[OK] $path"
    else
        echo "[MISS] $path"
    fi
done

echo

echo "------------------------------------------------------------"
echo "6. CI/CD"
echo "------------------------------------------------------------"

if [[ -d ".github/workflows" ]]; then
    find .github/workflows -maxdepth 1 -type f -print | sort
else
    echo "[MISS] .github/workflows"
fi

echo

echo "------------------------------------------------------------"
echo "7. Tests"
echo "------------------------------------------------------------"

if [[ -d tests ]]; then
    echo "Test files:"
    find tests -type f \
        \( -name 'test_*.py' -o -name '*_test.py' \) \
        | sort \
        | wc -l
else
    echo "[MISS] tests/"
fi

echo

echo "------------------------------------------------------------"
echo "8. ADR inventory"
echo "------------------------------------------------------------"

if [[ -d docs ]]; then
    find docs -type f \
        \( -iname '*adr*' -o -iname '*decision*' \) \
        -print \
        | sort
fi

echo

echo "------------------------------------------------------------"
echo "9. Current Compose runtime"
echo "------------------------------------------------------------"

if command -v docker >/dev/null 2>&1; then
    docker compose ps || true
fi

echo

echo "------------------------------------------------------------"
echo "10. Current Git status"
echo "------------------------------------------------------------"

git status

echo
echo "============================================================"
echo " Baseline audit complete"
echo "============================================================"
