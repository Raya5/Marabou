#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run subset experiment suite.
# ============================================================

echo "[run_subset] start $(date)"

bash "experiments/scripts/run_subset_robustness.sh"
bash "experiments/scripts/run_subset_explainability.sh"

echo "[run_subset] done $(date)"
echo "[run_subset] Successfully completed subset run."
