#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J dome_bo
#SBATCH -o run/dome/dome_bo-%j.log
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export DOME_BO_DEADLINE=$(( $(date +%s) + 12*3600 ))
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u legacy/scripts/dome_bo_campaign.py run run/dome/bo_stage1
