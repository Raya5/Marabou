#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Submit full experiment suite (SLURM)
# ============================================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

mkdir -p slurm_outputs

echo "[run_full_slurm] submitting jobs..."

jid_expl=$(sbatch "$SCRIPT_DIR/run_full_explainability_slurm.sh" | awk '{print $4}')
echo "[run_full_slurm] explainability job id: $jid_expl"

jid_rob=$(sbatch "$SCRIPT_DIR/run_full_robustness_slurm.sh" | awk '{print $4}')
echo "[run_full_slurm] robustness job id: $jid_rob"

echo "[run_full_slurm] done"