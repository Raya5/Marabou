#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Full run for explainability
# ============================================================

NUM_POINTS=70

echo "[full-expl] start $(date)"
echo "[full-expl] running $NUM_POINTS points"

for ((i=0; i<NUM_POINTS; i++))
do
    echo "[full-expl] ($((i+1))/$NUM_POINTS) index=$i"
    python experiments/scripts/run_explainability.py --index "$i" --full
done


echo "[full-expl] done $(date)"
