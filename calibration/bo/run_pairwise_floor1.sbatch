#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -J pair_floor1
#SBATCH -o run/pairwise_floor1/slurm-%x-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE MPLBACKEND=Agg
export BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT PYTHONOPTIMIZE
pair=${1:?Expected pair 12, 23, or 13}
: "${PAIR_PLAN_SHA256:?Expected reviewed plan SHA-256}"
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/bo/pairwise_floor1.py "$pair" run
