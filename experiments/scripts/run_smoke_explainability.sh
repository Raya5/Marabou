#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Smoke test for explainability
# Runs 10 dataset points.
# ============================================================

SUCCESS_FILE="experiments/results/explainability/smoke/success.json"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

rm -f "$SUCCESS_FILE"

echo "[smoke-expl] start $(date)"
echo "[smoke-expl] running $NUM_POINTS points"

INDICES=(27 28 31 33 34 41 57 58 61 66)
NUM_POINTS=${#INDICES[@]}

counter=1
for i in "${INDICES[@]}"
do
    echo "[smoke-expl] ($counter/$NUM_POINTS) index=$i"
    python experiments/scripts/run_explainability.py --index "$i" --smoke
    ((counter++))
done

if [[ -f "$SUCCESS_FILE" ]]; then
    echo "[smoke-expl] success"
    cat "$SUCCESS_FILE"
else
    echo "[smoke-expl] ERROR: missing success marker: $SUCCESS_FILE"
    exit 1
fi

echo "[smoke-expl] done $(date)"