#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Subset run for explainability
# Runs a fixed subset of dataset points sequentially.
# ============================================================

INDICES=(27 28 31 33 34 41 57 58 61 66 71 83 86 104 105 127 150 155 168 185 190 194 205 214 215 218 226 227 240 256 259 260 279 280 289 305 309 316 337 354 360 361 364 374 385 387 393 404 409 419 491)
NUM_POINTS=${#INDICES[@]}

echo "[subset-expl] start $(date)"
echo "[subset-expl] running $NUM_POINTS points"

counter=1
for i in "${INDICES[@]}"
do
    echo "[subset-expl] ($counter/$NUM_POINTS) index=$i"
    python experiments/scripts/run_explainability.py --index "$i"
    ((counter++))
done

echo "[subset-expl] done $(date)"
