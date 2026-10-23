#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:10:00
#SBATCH -J transfer
#SBATCH -o run/transfer-slurm-%j.out
# usage: sbatch --export=ALL,RUN=run7 calibration/bo/run_transfer.sbatch   (RUN = run7 | run10)
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
exec timeout --signal=TERM --kill-after=5s 3900s /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python calibration/bo/run_transfer.py "$RUN"
