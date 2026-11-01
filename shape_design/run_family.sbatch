#!/usr/bin/env bash
# Usage: sbatch --array=0-(N-1) shape_design/run_family.sbatch STAGE MESH_MM TIMEOUT_S variant:k [variant:k ...]
# Wall time must exceed TIMEOUT_S by >= 30 min: default 2 h for 3600 s; pass `-t 03:00:00` for the 7200 s 1 mm check.
# Scheduler SIGTERM is converted by run_case.py into a recorded failure after killing the solver group.
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J shape209
#SBATCH -o run/shape_design/slurm-%A_%a.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
STAGE=$1; MESH=$2; TIMEOUT=$3; shift 3
CASES=("$@")
item=${CASES[$SLURM_ARRAY_TASK_ID]}
variant=${item%%:*}; k=${item#*:}
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE
export BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT
export MPLBACKEND=Agg
unset PYTHONOPTIMIZE
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python shape_design/run_case.py "$STAGE" "$variant" "$k" \
    --mesh-mm "$MESH" --timeout "$TIMEOUT"
