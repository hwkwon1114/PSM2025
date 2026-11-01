#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -J joint_s2A
#SBATCH -o run/joint_stage2/arm_A/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
mkdir -p run/joint_stage2/arm_A
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE MPLBACKEND=Agg
unset PYTHONOPTIMIZE
echo "Stage2 arm A $(date -Iseconds) job=${SLURM_JOB_ID}"
export JOINT_SPEC=/gpfs/projects/p33082/PSM2025-zigzag/calibration/bo/stage2_arm_A.json
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/bo/run_joint_run6_run10.py run
