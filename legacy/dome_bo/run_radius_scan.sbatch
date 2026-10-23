#!/bin/bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --array=0-35%6
#SBATCH -J radius_scan
#SBATCH -o run/dome_square/mesh_study/logs/radius_%A_%a.out
#SBATCH -e run/dome_square/mesh_study/logs/radius_%A_%a.err
# Curvature-radius scan of retained mesh-study states (no solves). 1 mm cells first.
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CELLS=()
for r in .005 .01 .02; do for a in 30 31 21 32 27 19; do for R in 25 30; do CELLS+=("$a $r $R"); done; done; done
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python legacy/dome_bo/radius_scan.py --cell ${CELLS[$SLURM_ARRAY_TASK_ID]}
