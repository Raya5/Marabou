#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

echo "[run_smoke] start $(date)"

bash "$SCRIPT_DIR/run_smoke_robustness.sh"
bash "$SCRIPT_DIR/run_smoke_explainability.sh"

echo "[run_smoke] done $(date)"