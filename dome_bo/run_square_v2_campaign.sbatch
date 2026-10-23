#!/bin/bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 36:00:00
#SBATCH -J square_dome_bo_v2
#SBATCH -o run/dome_square_v2/logs/slurm-%j.out
#SBATCH -e run/dome_square_v2/logs/slurm-%j.err
# Square-sheet dome search v2 (docs/dome_bo/square_bo_v2_protocol.md).
# One 32-CPU/128 GB allocation; the controller runs four 8-CPU/32 GB solver steps at a time.
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
END_EPOCH=$(( $(date +%s) + 36*3600 - 300 ))
/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python dome_bo/square_v2_campaign.py \
    --root run/dome_square_v2 --end-epoch "${END_EPOCH}" \
    --launcher "srun -N1 -n1 -c8 --mem=32G --exact"
