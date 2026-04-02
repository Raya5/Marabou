#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Smoke test for robustness
# Runs NUM_POINTS dataset points sequentially.
# ============================================================

NUM_POINTS=10

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "[smoke-rob] start $(date)"
echo "[smoke-rob] running $NUM_POINTS points"

for ((i=0; i<NUM_POINTS; i++))
do
    echo "[smoke-rob] ($((i+1))/$NUM_POINTS) index=$i"
    python experiments/scripts/run_robustness.py \
        --index "$i" \
        --smoke
done

echo "[smoke-rob] done $(date)"