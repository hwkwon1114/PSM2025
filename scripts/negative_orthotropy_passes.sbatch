#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --job-name=negortho_passes
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/forward_model_diagnostics/negative_ortho_passes_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
cd /gpfs/projects/p33082/PSM2025-zigzag
sha256sum -c run/forward_model_diagnostics/absolute_stop_audit/verified_binary.sha256
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_DYNAMIC=false MPLBACKEND=Agg
status=0
python python/run_negative_orthotropy_passes.py || status=$?
# Preserve and plot accepted partial endpoints even if a later pass fails.
python python/plot_negative_orthotropy_passes.py
exit "$status"
