#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J dome_prop_matched
#SBATCH -o run/dome/proposed_matched-%j.log
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python legacy/scripts/run_dome.py \
  --sequence run/dome/proposed_matched_inputs/recipe.json \
  --output run/dome/proposed_matched_solve
