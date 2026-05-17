#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Smoke test for robustness
# Runs NUM_POINTS dataset points.
# ============================================================

INDICES=(0 1)
NUM_POINTS=${#INDICES[@]}

echo "[smoke-rob] start $(date)"

for ((j=0; j<NUM_POINTS; j++))
do
    i=${INDICES[$j]}
    echo "[smoke-rob] index=$i"
    python experiments/scripts/run_robustness.py --index "$i" --smoke
done
echo "[smoke-rob] success"
echo "[smoke-rob] done $(date)"