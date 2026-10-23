#!/bin/bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH -J overlap_seq
#SBATCH -o run/overlapping_sequence/logs/job-%j.log
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
unset PYTHONOPTIMIZE

case_name=${1:?case required: ab_block, ba_block, ab_alternate, ba_alternate, a_only, b_only}
mode=${2:?mode required: every_cycle, final_only}
variant=${3:-base}

mkdir -p run/overlapping_sequence/logs

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u legacy/scripts/run_overlapping_sequence.py \
  --case "$case_name" --mode "$mode" --variant "$variant" \
  --output "run/overlapping_sequence/matrix/${case_name}_${mode}_${variant}"
