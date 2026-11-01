#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 02:40:00
#SBATCH -J run2_start_diag
#SBATCH -o run/diagnostics/run2_initial_state_a0084/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 MPLBACKEND=Agg
ulimit -c 0
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/analysis/run2_initial_state_diagnostic.py run
