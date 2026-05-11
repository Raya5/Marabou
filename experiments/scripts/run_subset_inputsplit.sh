#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Subset run for input splitting
# Runs NUM_POINTS dataset points.
# ============================================================

NUM_POINTS=5

echo "[subset-inp] start $(date)"
echo "[subset-inp] running $NUM_POINTS points"


python experiments/scripts/run_experiment_inputsplit.py \
    --benchmarks-dir experiments/data/benchmarks \
    --timeout 5 --timeout-factor 1.5 --max-depth 16 \
    --subset 

echo "[subset-inp] done $(date)"