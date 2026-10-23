#!/bin/bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J square_verify30
#SBATCH -o run/dome_square/logs/verify-%j.out
#SBATCH -e run/dome_square/logs/verify-%j.err
# One fresh fine-mesh solve of the best frozen search-objective candidate; no retry.
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
printf '%s  %s\n' 522633cff2d7ff14b392c0ddd58a80df5049ed9648f8f6de258ff7d1801c4841 run/dome_square/frozen/shell | sha256sum -c -
OUT=run/dome_square/verification/attempt_030
mkdir -p run/dome_square/verification
mkdir "$OUT"
cp run/dome_square/campaign/attempts/attempt_030/request.json "$OUT/request.json"
timeout --signal=TERM --kill-after=30s 4500s \
    /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python dome_bo/square_attempt.py \
    --attempt-dir "$OUT" --verify --threads 8
