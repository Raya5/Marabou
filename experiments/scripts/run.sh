#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run experiment suite.
# ============================================================

# Default mode is "subset"
MODE="subset"

# Check for flags
if [[ $# -gt 0 ]]; then
    case "$1" in
        --full|full)
            MODE="full"
            ;;
        --subset|subset)
            MODE="subset"
            ;;
        *)
            echo "Usage: $0 [--subset | --full]"
            exit 1
            ;;
    esac
fi

echo "[run_$MODE] start $(date)"

if [[ "$MODE" == "subset" ]]; then
    bash "experiments/scripts/run_subset_robustness.sh"
    bash "experiments/scripts/run_subset_explainability.sh"
else
    bash "experiments/scripts/run_full_robustness.sh"
    bash "experiments/scripts/run_full_explainability.sh"
fi

echo "[run_$MODE] done $(date)"
echo "[run_$MODE] Successfully completed $MODE run."