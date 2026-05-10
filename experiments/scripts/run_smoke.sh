#!/usr/bin/env bash
set -euo pipefail

echo "[run_smoke] start $(date)"

bash "experiments/scripts/run_smoke_robustness.sh"
bash "experiments/scripts/run_smoke_inputsplit.sh"
bash "experiments/scripts/run_smoke_explainability.sh"

echo "[run_smoke] done $(date)"