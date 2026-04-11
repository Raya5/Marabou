#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Subset run for robustness
# Runs NUM_POINTS dataset points.
# ============================================================

NUM_POINTS=100

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "[subset-rob] start $(date)"
echo "[subset-rob] running $NUM_POINTS points"

for ((i=0; i<NUM_POINTS; i++))
do
    echo "[subset-rob] ($((i+1))/$NUM_POINTS) index=$i"
    python experiments/scripts/run_robustness.py --index "$i"
done

echo "[subset-rob] done $(date)"