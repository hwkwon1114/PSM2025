#!/usr/bin/env bash
# Forward prediction of Run 3 (30 passes) with Attempt 209 model on compute node
#SBATCH -A p33082
#SBATCH -p short
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J hw_run3
#SBATCH -o run/shape_design/hw_run3/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE
export BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT
export MPLBACKEND=Agg
unset PYTHONOPTIMIZE
PY=/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python

echo "=== Running Run 3 Solve ==="
$PY shape_design/run_hw3.py solve

echo "=== Running Scan Comparison ==="
$PY shape_design/compare_hw1.py --solve run/shape_design/hw_run3/solve_mesh2 --scan shape_design/data/Run_3.ply

echo "=== Done ==="

