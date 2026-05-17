#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Subset run for explainability
# Runs a fixed subset of dataset points sequentially.
# ============================================================

NUM_POINTS=5

echo "[subset-expl] start $(date)"
echo "[subset-expl] running $NUM_POINTS points"

for ((i=0; i<NUM_POINTS; i++))
do
    echo "[subset-expl] ($((i+1))/$NUM_POINTS) index=$i"
    python experiments/scripts/run_explainability.py --index "$i" --subset
done


echo "[subset-expl] done $(date)"
