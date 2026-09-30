#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --job-name=three_retry
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/forward_model_diagnostics/three_retry_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OMP_DYNAMIC=false
python python/run_split_three_passes.py --index 4 --max-iter 1000000 --timeout 85800
