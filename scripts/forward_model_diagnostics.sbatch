#!/bin/bash
#SBATCH --job-name=fwd_diag
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/forward_model_diagnostics/slurm_%j.out
#SBATCH --error=/gpfs/projects/p33082/PSM2025-zigzag/run/forward_model_diagnostics/slurm_%j.out

# Serial forward-model diagnostics: pinned upstream b165fc52 vs current on the
# shared nested and diversity families, plus the current-only frozen-final-
# loading initialization family. Cases run one after another in this task.
#
# Selection (all optional, space-separated lists):
#   FMD_IMPL="current upstream"      implementations to run
#   FMD_FAMILY="nested diversity"    families to run (nested diversity frozen)
#   FMD_CASE="iso_top_full ..."      individual case names
#   FMD_TIMEOUT=1800                 per-case wall-clock limit, s
#   FMD_EXTRA_ARGS="--force"         passed through verbatim
#
# Examples:
#   sbatch scripts/forward_model_diagnostics.sbatch                     # all
#   FMD_FAMILY=diversity sbatch scripts/forward_model_diagnostics.sbatch
#   FMD_IMPL=upstream FMD_FAMILY="nested diversity" \
#       sbatch scripts/forward_model_diagnostics.sbatch
#   FMD_FAMILY=frozen FMD_TIMEOUT=3600 \
#       sbatch scripts/forward_model_diagnostics.sbatch
#
# The nested and frozen families reuse the 6-cycle crown_nested_broad_to_central
# recipe, which cost 858 s in run/solver_characteristics with the current binary
# at 8 threads; budget accordingly and keep FMD_TIMEOUT above that. The five
# diversity cases are single-cycle and are the compact few-minutes set.

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u

repo_root=/gpfs/projects/p33082/PSM2025-zigzag
cd "$repo_root"
mkdir -p run/forward_model_diagnostics

export OMP_DYNAMIC=false
cpus="${SLURM_CPUS_PER_TASK:-8}"

runner=(python python/run_forward_model_diagnostics.py)
runner+=(--timeout "${FMD_TIMEOUT:-1800}")
runner+=(--threads "$cpus")
for impl in ${FMD_IMPL:-}; do
  runner+=(--implementation "$impl")
done
for family in ${FMD_FAMILY:-}; do
  runner+=(--family "$family")
done
for case_name in ${FMD_CASE:-}; do
  runner+=(--case "$case_name")
done
# shellcheck disable=SC2206
extra=(${FMD_EXTRA_ARGS:-})
runner+=("${extra[@]+"${extra[@]}"}")

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${cpus}"
echo "cmd: ${runner[*]}"
status=0
"${runner[@]}" || status=$?

python python/analyze_forward_model_diagnostics.py || status=1
if [[ "${FMD_FAMILY:-}" == "crosswheel" && "$status" == 0 ]]; then
  python python/analyze_crosswheeling.py || status=1
fi
exit "$status"
