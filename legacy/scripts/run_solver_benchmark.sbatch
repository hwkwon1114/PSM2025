#!/bin/bash
#SBATCH -A p33082
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -J solver_benchmark
#SBATCH -o run/solver_benchmark/logs/slurm-%j.out
#SBATCH -e run/solver_benchmark/logs/slurm-%j.err

set -euo pipefail

cd /gpfs/projects/p33082/PSM2025-zigzag
mkdir -p run/solver_benchmark/logs

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

PYTHON="/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python"

echo "=== Quest Solver Benchmark Execution ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-local}"

CASE="${1:-all}"
POLICY="${2:-cutback}"
FORCE="${3:---force}"

echo "Executing benchmark for case: ${CASE}, policy: ${POLICY}, force: ${FORCE}"

${PYTHON} legacy/scripts/run_solver_benchmark.py --case "${CASE}" --policy "${POLICY}" ${FORCE}

echo "Benchmark run finished at $(date)"
