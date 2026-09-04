#!/bin/bash
#SBATCH --job-name=eq_cont20
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/equilibrium_continuation_study/slurm_%j.out
#SBATCH --error=/gpfs/projects/p33082/PSM2025-zigzag/run/equilibrium_continuation_study/slurm_%j.out

set -euo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

repo_root=/gpfs/projects/p33082/PSM2025-zigzag
cd "$repo_root"
python python/run_equilibrium_continuation_study.py --tier all --timeout 39600
