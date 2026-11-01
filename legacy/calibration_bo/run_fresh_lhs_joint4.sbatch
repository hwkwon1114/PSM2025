#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -J joint_lhs4
#SBATCH -o run/fresh_lhs_joint4/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE MPLBACKEND=Agg
unset PYTHONOPTIMIZE
export JOINT_SPEC=/gpfs/projects/p33082/PSM2025-zigzag/calibration/bo/fresh_lhs_joint4.json
echo "Fresh LHS joint4 $(date -Iseconds) job=${SLURM_JOB_ID}"
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/bo/run_joint_run6_run10.py run
