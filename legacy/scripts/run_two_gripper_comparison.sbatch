#!/bin/bash
#SBATCH --job-name=two_gripper_release
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=run/two_gripper_comparison/slurm-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
mode_args=()
suffix=''
if [[ "${1:-}" == --final-only ]]; then
    mode_args+=(--never-clamped --final-only)
    suffix='_final_only'
    shift
    # Submit this mode with sbatch --time=00:30:00 (one final solve).
elif [[ "${1:-}" == --never-clamped ]]; then
    mode_args+=(--never-clamped)
    suffix='_never_clamped'
    shift
fi
if [[ $# -lt 1 || $# -gt 3 || ( "$1" != run6 && "$1" != run9 ) ]]; then
    echo 'Usage: sbatch legacy/scripts/run_two_gripper_comparison.sbatch [--never-clamped|--final-only] run6|run9 [new_output_dir [accepted_checkpoint]]' >&2
    exit 2
fi
# Conda compiler activation hooks read optional unset variables.
set +u
source /gpfs/home/pxl1051/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=8 OMP_THREAD_LIMIT=8 OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
args=("${mode_args[@]}" --case "$1" --output "${2:-run/two_gripper_comparison/$1$suffix}")
if [[ $# -ge 3 ]]; then args+=(--resume "$3"); fi
python legacy/scripts/run_two_gripper_comparison.py "${args[@]}"
