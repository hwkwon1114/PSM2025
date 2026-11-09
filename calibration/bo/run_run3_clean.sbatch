#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J run3_clean
#SBATCH -o run/run3_clean_24lhs/slurm-%j.out

set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
mkdir -p run/run3_clean_24lhs

export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE MPLBACKEND=Agg
export BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT PYTHONOPTIMIZE

exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/bo/clean_calibration_runner.py calibration/bo/run3_clean_24lhs.json run
