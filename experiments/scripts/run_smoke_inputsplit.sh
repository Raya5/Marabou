#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Smoke run for input splitting
# Runs NUM_POINTS dataset points.
# ============================================================


echo "[smoke-inp] start $(date)"

python experiments/scripts/run_experiment_inputsplit.py \
    --benchmarks-dir experiments/data/benchmarks \
    --timeout 5 --timeout-factor 1.5 --max-depth 16 \
    --smoke 

echo "[smoke-inp] done $(date)"