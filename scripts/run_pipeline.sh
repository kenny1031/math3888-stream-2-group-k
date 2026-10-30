#!/usr/bin/env bash

# ============================================================
# MATH3888 — HLJ1 Protein Network Analysis
#
# Usage:
#   bash scripts/run_pipeline.sh network
#   bash scripts/run_pipeline.sh cadtp
#   bash scripts/run_pipeline.sh validate
#   bash scripts/run_pipeline.sh evaluate
#   bash scripts/run_pipeline.sh plot
#   bash scripts/run_pipeline.sh all
#   bash scripts/run_pipeline.sh test
# ============================================================

set -euo pipefail

# Resolve repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

# Select Python interpreter
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON="${PYTHON_BIN}"
elif [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "Error: Python interpreter not found."
    exit 1
fi

# Parse stage
STAGE="${1:-}"

if [[ -z "${STAGE}" ]]; then
    echo "Usage: bash scripts/run_pipeline.sh <stage>"
    echo ""
    echo "Available stages:"
    echo "  network    Construct and validate networks"
    echo "  cadtp      Generate CADTP training rankings"
    echo "  validate   Run held-out Louvain validation"
    echo "  evaluate   Run GO biological evaluation"
    echo "  plot       Generate research figures"
    echo "  all        Run the complete current pipeline"
    echo "  test       Run all unit and regression tests"
    exit 1
fi

shift

# Run
export PYTHONHASHSEED=0
export MPLBACKEND=Agg

echo "=============================================="
echo "MATH3888 — HLJ1 Protein Network Analysis"
echo "Stage: ${STAGE}"
echo "Python: ${PYTHON}"
echo "=============================================="

case "${STAGE}" in
    network|cadtp|validate|evaluate|plot|all)
        "${PYTHON}" -m src.pipeline \
            --stage "${STAGE}" "$@"
        ;;
    test)
        "${PYTHON}" -m pytest tests/ -q "$@"
        ;;
    *)
        echo "Error: Unknown stage '${STAGE}'"
        exit 1
        ;;
esac

echo ""
echo "Stage '${STAGE}' completed successfully!"
