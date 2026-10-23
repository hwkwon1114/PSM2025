#!/bin/bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 21:00:00
#SBATCH -J sequence_continuation
#SBATCH -o run/sequence_order/checks/job-%j.log
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
unset PYTHONOPTIMIZE
case_name=${1:?case required}
mode=${2:?mode required}
variant=${3:-base}
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u legacy/scripts/run_sequence_order_continuation.py \
 --case "$case_name" --mode "$mode" --variant "$variant" \
 --output "run/sequence_order/continuation_matrix/${case_name}_${mode}_${variant}"
