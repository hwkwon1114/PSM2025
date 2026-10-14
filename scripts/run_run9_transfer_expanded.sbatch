#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 01:10:00
#SBATCH -J run9_trans_exp
#SBATCH -o run/run9_transfer_expanded/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
mkdir -p run/run9_transfer_expanded
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8 NUMEXPR_NUM_THREADS=8
: "${TRANSFER_PACKET_SHA256:?Reviewed transfer packet SHA256 required}"
echo "Frozen run6-to-run9 expanded transfer; $(date -Iseconds), job=${SLURM_JOB_ID:-manual}, host=$(hostname)"
exec timeout --signal=TERM --kill-after=5s 3900s /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python scripts/run_run9_transfer_expanded.py --packet-sha "$TRANSFER_PACKET_SHA256"
