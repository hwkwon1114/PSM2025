#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=split_growth
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --array=0-17
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/forward_model_diagnostics/split_growth_%A_%a.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OMP_DYNAMIC=false
python python/run_split_growth_sweep.py --index "$SLURM_ARRAY_TASK_ID"
