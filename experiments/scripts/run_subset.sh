#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run subset experiment suite.
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "[run_subset] start $(date)"

bash "$SCRIPT_DIR/run_subset_robustness.sh"
bash "$SCRIPT_DIR/run_subset_explainability.sh"

echo "[run_subset] done $(date)"
echo "[run_subset] Successfully completed subset run."
