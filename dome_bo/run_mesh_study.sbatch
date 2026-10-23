#!/bin/bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:30:00
#SBATCH --array=0-16%4
#SBATCH -J square_mesh
#SBATCH -o run/dome_square/mesh_study/logs/%A_%a.out
#SBATCH -e run/dome_square/mesh_study/logs/%A_%a.err
# One fresh solve per array task (no retry); at most 4 concurrent.
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
printf '%s  %s\n' 522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841 run/dome_square/frozen/shell | sha256sum -c -
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python dome_bo/mesh_study.py "$SLURM_ARRAY_TASK_ID"
