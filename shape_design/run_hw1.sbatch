#!/usr/bin/env bash
# Card REPLICATE-HW1: one frozen Attempt 209 solve (2 mm, 3600 s limit), then the solver-free scan comparison.
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J hw_run1
#SBATCH -o run/shape_design/hw_run1/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE
export BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT
export MPLBACKEND=Agg
unset PYTHONOPTIMIZE
PY=/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python
$PY shape_design/run_hw1.py solve
$PY shape_design/compare_hw1.py
