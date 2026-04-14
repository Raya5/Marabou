#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Submit full experiment suite (SLURM)
# ============================================================
mkdir -p slurm_outputs

echo "[run_full_slurm] submitting jobs..."

jid_expl=$(sbatch "experiments/scripts/run_full_explainability_slurm.sh" | awk '{print $4}')
echo "[run_full_slurm] explainability job id: $jid_expl"

jid_rob=$(sbatch "experiments/scripts/run_full_robustness_slurm.sh" | awk '{print $4}')
echo "[run_full_slurm] robustness job id: $jid_rob"

echo "[run_full_slurm] done"