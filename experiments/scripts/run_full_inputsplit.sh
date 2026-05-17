#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Full run for input splitting
# Runs NUM_POINTS dataset points.
# ============================================================

echo "[full-inp] start $(date)"

python experiments/scripts/run_experiment_inputsplit.py \
    --benchmarks-dir experiments/data/benchmarks \
    --timeout 5 --timeout-factor 1.5 --max-depth 16 \
    --full 

echo "[full-inp] done $(date)"