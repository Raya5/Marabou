#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Smoke test for robustness
# Runs NUM_POINTS dataset points.
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "[smoke-rob] start $(date)"
echo "[smoke-rob] running $NUM_POINTS points"

INDICES=(0 1 2 3 4 5 6 7 8 9)
NUM_POINTS=${#INDICES[@]}

for ((j=0; j<NUM_POINTS; j++))
do
    i=${INDICES[$j]}
    echo "[smoke-rob] ($((j+1))/$NUM_POINTS) index=$i"
    python experiments/scripts/run_robustness.py --index "$i" --smoke
done

echo "[smoke-rob] done $(date)"