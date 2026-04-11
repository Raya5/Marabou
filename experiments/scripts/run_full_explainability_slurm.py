#!/bin/bash
#SBATCH --job-name=iv_explainability
#SBATCH --comment="IV explainability"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4gb
#SBATCH --time=6:00:00
#SBATCH --output=slurm_outputs/slurm__%A_%a.log
#SBATCH --array=0-999%50

set -euo pipefail

# ============================================================
# SLURM array run for explainability
# Each array task runs one dataset index.
# ============================================================

INDEX="${SLURM_ARRAY_TASK_ID}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_DIR"

mkdir -p slurm_outputs

echo "[slurm-expl] start $(date)"
echo "[slurm-expl] job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID index=$INDEX"

python experiments/scripts/run_explainability.py --index "$INDEX"

echo "[slurm-expl] done $(date)"