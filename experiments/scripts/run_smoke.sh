#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run subset experiment suite on 10 points for each use case.
# ============================================================

SCRIPT_DIR="experiments/scripts/"

echo "[run_smoke] start $(date)"

bash "$SCRIPT_DIR/run_smoke_robustness.sh"
bash "$SCRIPT_DIR/run_smoke_explainability.sh"

echo "[run_smoke] done $(date)"

echo "[run_smoke] Successfully completed smoke test."

# this is the final smoke test to get the time and submit the code for the smoke.
# /cs/labs/guykatz/rayae/slurmy/run.sh experiments/scripts/run_smoke.sh
# Job 29660142 output is in /cs/labs/guykatz/rayae/slurmy/mar2026/outputs/o__29660142.log