#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Smoke test for explainability
# Runs 10 dataset points.
# ============================================================

SUCCESS_FILE="experiments/results/explainability/smoke/success.json"

rm -f "$SUCCESS_FILE"

INDICES=(27 28)
NUM_POINTS=${#INDICES[@]}

echo "[smoke-expl] start $(date)"

counter=1
for i in "${INDICES[@]}"
do
    echo "[smoke-expl] index=$i"
    python experiments/scripts/run_explainability.py --index "$i" --smoke
    ((counter++))
    if [[ -f "$SUCCESS_FILE" ]]; then
        break
    fi
done

if [[ -f "$SUCCESS_FILE" ]]; then
    echo "[smoke-expl] success"
else
    echo "[smoke-expl] ERROR: explainability smoke test failed. Expected success file not found."
    exit 1
fi

echo "[smoke-expl] done $(date)"