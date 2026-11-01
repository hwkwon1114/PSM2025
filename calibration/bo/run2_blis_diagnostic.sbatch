#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 02:30:00
#SBATCH -J run2_blis_diag
#SBATCH -o run/diagnostics/run2_blis_a0084_r1/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONHASHSEED=0
unset BLIS_NUM_THREADS BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT
ulimit -c 0
# One bound step gives both arms the same node and allocated CPU affinity.
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores bash -s <<'STEP'
set -euo pipefail
PY=/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python
SCRIPT=calibration/analysis/run2_blis_diagnostic.py
env -u BLIS_NUM_THREADS "$PY" -u "$SCRIPT" run unset
env BLIS_NUM_THREADS=1 "$PY" -u "$SCRIPT" run one
env -u BLIS_NUM_THREADS "$PY" -u "$SCRIPT" summarize
STEP
