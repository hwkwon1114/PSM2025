#!/bin/bash
#SBATCH --job-name=dome_s1
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-31%5
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/same_side_dome_search/stage1/slurm_%A_%a.out
#SBATCH --error=/gpfs/projects/p33082/PSM2025-zigzag/run/same_side_dome_search/stage1/slurm_%A_%a.out

set -o pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

repo_root=/gpfs/projects/p33082/PSM2025-zigzag
output_root=$repo_root/run/same_side_dome_search/stage1

python "$repo_root/python/same_side_dome_search.py" run-task \
  "$SLURM_ARRAY_TASK_ID" "$output_root" \
  --repo-root "$repo_root" \
  --resolution 0.03 \
  --tolerance 1e-12 \
  --timeout 14000
