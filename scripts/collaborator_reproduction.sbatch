#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=crown_repro
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --array=0-2
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/forward_model_diagnostics/crown_repro_%A_%a.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=8 OMP_DYNAMIC=false
case "$SLURM_ARRAY_TASK_ID" in
  0) implementation=current; iterations=200000 ;;
  1) implementation=upstream; iterations=50000 ;;
  2) implementation=upstream; iterations=200000 ;;
  *) exit 2 ;;
esac
python python/run_collaborator_crown.py --implementation "$implementation" --max-iter "$iterations"
