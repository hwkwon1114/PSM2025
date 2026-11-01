#!/usr/bin/env bash
#SBATCH -A p33082
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -J joint_pareto4
#SBATCH -o run/multiobjective_joint4/slurm-%j.out
set -euo pipefail
cd /gpfs/projects/p33082/PSM2025-zigzag
export OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE MPLBACKEND=Agg
export BLIS_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset BLIS_JC_NT BLIS_PC_NT BLIS_IC_NT BLIS_JR_NT BLIS_IR_NT PYTHONOPTIMIZE
export JOINT_SPEC=/gpfs/projects/p33082/PSM2025-zigzag/calibration/bo/multiobjective_joint4.json
echo "Run 1/3 Pareto continuation $(date -Iseconds) job=${SLURM_JOB_ID}"
exec /gpfs/home/pxl1051/miniforge/envs/DHD/bin/python -u calibration/bo/joint_calibration.py run
