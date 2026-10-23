#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:10:00
#SBATCH -J run7_transfer
#SBATCH -o run/run7_transfer/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
mkdir -p run/run7_transfer
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
exec timeout --signal=TERM --kill-after=5s 3900s /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python calibration/bo/run_run7_transfer.py
