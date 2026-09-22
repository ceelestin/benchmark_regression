#!/bin/bash
# ============================================================================
# Jean-Zay SLURM job array for the redundancy-score DERIVATION experiment, CLASSIFICATION analog (configs/derivation_redundancy_levers_classification_*.yml).
#
# Each array task runs benchopt on ONE seed-chunk config (see make_chunks.py)
# on ONE full cpu_p1 node (40 cores), with 40 single-core benchopt workers.
# Each task writes its own parquet into $SCRATCH/ranking_outputs/.
#
# EDIT the three lines marked  <<< EDIT  before submitting:
#   1. --account     : your IDRIS CPU allocation (run `idracctinfo` to find it)
#   2. --array range : 0-(Ntasks-1); make_chunks.py prints Ntasks
#   3. the `source`  : activate your env (venv or conda)
# ============================================================================
#SBATCH --job-name=deriv_clf
#SBATCH --partition=cpu_p1
#SBATCH --qos=qos_cpu-t3                 # <= 20 h walltime (default CPU QoS)
#SBATCH --account=sth@cpu                # <<< EDIT: your-project@cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40               # a full cpu_p1 node
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00                  # ~1.5 h expected per K=200 task, 4x margin
#SBATCH --array=0-22%23                  # 2 x 10 K=200 chunks (3 seeds) + 3 fixed-split chunks (10 seeds)
#SBATCH --output=logs/rank_%A_%a.out
#SBATCH --error=logs/rank_%A_%a.err

set -euo pipefail

module purge
# --- activate your environment  <<< EDIT to match your setup --------------
source "$WORK/venvs/benchcv/bin/activate"     # venv
# conda activate "$WORK/envs/benchcv"         # ... or conda

# One benchopt worker per physical core, each scikit-learn fit single-threaded
# (avoids 40x oversubscription of the BLAS/OpenMP thread pools).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$WORK/benchmark_regression"

MANIFEST=configs/chunks_classification/manifest.txt
CFG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
NAME=$(basename "$CFG" .yml)

if [[ -z "$CFG" ]]; then
    echo "No config at manifest line $((SLURM_ARRAY_TASK_ID + 1)) — array range too large?"
    exit 1
fi

echo "[$(date '+%F %T')] task ${SLURM_ARRAY_TASK_ID} | ${CFG} | node ${SLURMD_NODENAME}"

# benchopt runs solvers in the *current* env; `outputs/` is symlinked to
# $SCRATCH (see JEANZAY_README.md), so `--output $NAME` lands the parquet there (benchopt >= 1.10: `-o` is the objective filter).
benchopt run . --config "$CFG" \
    --no-plot --no-cache --no-timeout \
    -j 40 --output "$NAME"

echo "[$(date '+%F %T')] done -> outputs/${NAME}.parquet"
