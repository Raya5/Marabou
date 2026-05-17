#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Full run for robustness
# ============================================================

NUM_POINTS=185

echo "[full-rob] start $(date)"
echo "[full-rob] running $NUM_POINTS points"

for ((i=0; i<NUM_POINTS; i++))
do
    echo "[full-rob] ($((i+1))/$NUM_POINTS) index=$i"
    python experiments/scripts/run_robustness.py --index "$i" --full
done

echo "[full-rob] done $(date)"