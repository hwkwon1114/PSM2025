#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=segment_diag
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/forward_model_diagnostics/segmentation_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OMP_DYNAMIC=false
python python/run_segmentation_experiment.py
python python/analyze_forward_model_diagnostics.py
