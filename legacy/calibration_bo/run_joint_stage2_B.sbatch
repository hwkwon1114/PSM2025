#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -J joint_s2B
#SBATCH -o run/joint_stage2/arm_B/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
mkdir -p run/joint_stage2/arm_B
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE MPLBACKEND=Agg
unset PYTHONOPTIMIZE
echo "Stage2 arm B $(date -Iseconds) job=${SLURM_JOB_ID}"
export JOINT_SPEC=/gpfs/projects/p33082/PSM2025-zigzag/calibration/bo/stage2_arm_B.json
OMP_NUM_THREADS=8 /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/analysis/complete_armB_a0013.py >> run/joint_stage2/arm_B/logs/a0013.log 2>&1
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/bo/run_joint_run6_run10.py run
