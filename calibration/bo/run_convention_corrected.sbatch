#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J convention
#SBATCH -o run/convention_corrected/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
case "$1" in run6|run9|run10) ;; *) exit 2;; esac
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE
export BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
unset PYTHONOPTIMIZE
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python calibration/bo/run_convention_corrected.py "$1"
