#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --no-requeue
#SBATCH -J run6_bo_recover
#SBATCH -o run/run6_expanded_bo_continue20_recovery/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8
: "${PACKET_SHA256:?Reviewed recovery packet digest required}"
PYTHON=/gpfs/home/pxl1051/miniforge/envs/DHD/bin/python
echo "Run6 remaining9 waves $(date -Iseconds) job=${SLURM_JOB_ID}"
exec "$PYTHON" -u scripts/run_run6_expanded_bo_recovery.py run --packet-sha256 "$PACKET_SHA256"
