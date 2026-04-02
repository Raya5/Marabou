#!/bin/bash
#SBATCH --job-name=iv_robustness
#SBATCH --comment="IV robustness"
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=6:00:00
#SBATCH --output=slurm_outputs/slurm__%A_%a.log
#SBATCH --array=0-999%50

set -euo pipefail

# ============================================================
# SLURM array run for robustness
# Each array task runs one dataset index.
# ============================================================

INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"

echo "[slurm-rob] start $(date)"
echo "[slurm-rob] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID index=$INDEX"

python experiments/scripts/run_robustness.py --index "$INDEX"

echo "[slurm-rob] done $(date)"