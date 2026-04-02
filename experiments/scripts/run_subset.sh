#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run subset experiment suite.
# ============================================================

SCRIPT_DIR="experiments/scripts/"

echo "[run_subset] start $(date)"

bash "$SCRIPT_DIR/run_subset_robustness.sh"
bash "$SCRIPT_DIR/run_subset_explainability.sh"

echo "[run_subset] done $(date)"
echo "[run_subset] Successfully completed subset run."

# Last subset run to get the time and submit the code for the subset.
# /cs/labs/guykatz/rayae/slurmy/run.sh experiments/scripts/run_subset.sh
# Job 29660271 output is in /cs/labs/guykatz/rayae/slurmy/mar2026/outputs/o__29660271.log