#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -J joint_r6r10
#SBATCH -o run/joint_run6_run10/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
mkdir -p run/joint_run6_run10
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE MPLBACKEND=Agg
unset PYTHONOPTIMIZE
echo "Joint Run6+Run10 calibration $(date -Iseconds) job=${SLURM_JOB_ID}"
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/bo/run_joint_run6_run10.py run
