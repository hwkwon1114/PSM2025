#!/bin/bash
#SBATCH --job-name=solver_char
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/solver_characteristics/slurm_%j.out
#SBATCH --error=/gpfs/projects/p33082/PSM2025-zigzag/run/solver_characteristics/slurm_%j.out

# Serial solver-characteristics study. Every case runs one after another inside
# this single task; the runner also holds an exclusive lock, so two concurrent
# submissions cannot interleave solver runs.
#
# Selection (all optional, families are separate arguments):
#   SC_FAMILY="path_dependence shrinking_crown"   space-separated family list
#   SC_CASE="order_forward_multiplicative ..."    space-separated case list
#   SC_TIMEOUT=10800                              per-case wall-clock limit, s
#   SC_THREADS=8                                  fixed thread count for the
#                                                 repeatability replica group
#   SC_REPLICAS=5                                 replicas at SC_THREADS (>=3)
#   SC_SINGLE_REPLICAS=3                          replicas at one thread (>=3)
#   SC_ENABLE_PRESCRIBED=1                        run the prescribed family
#   SC_PRESCRIBED_FLAGS="k=v k=v"                 extra -prescribed_* flags
#   SC_EXTRA_ARGS="--force"                       passed through verbatim
#
# Examples:
#   sbatch scripts/solver_characteristics.sbatch                       # all
#   SC_FAMILY=path_dependence sbatch scripts/solver_characteristics.sbatch
#   SC_FAMILY="shrinking_crown" sbatch --time=06:00:00 \
#       scripts/solver_characteristics.sbatch
#   SC_FAMILY=hlbfgs_reproducibility SC_REPLICAS=5 \
#       sbatch scripts/solver_characteristics.sbatch
#   SC_ENABLE_PRESCRIBED=1 SC_FAMILY=prescribed_deformation \
#       sbatch scripts/solver_characteristics.sbatch

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u

repo_root=/gpfs/projects/p33082/PSM2025-zigzag
cd "$repo_root"
mkdir -p run/solver_characteristics

# The runner sets OMP_NUM_THREADS per case (fixed-thread and single-thread
# replicas), so it must not be pinned here.
export OMP_DYNAMIC=false

cpus="${SLURM_CPUS_PER_TASK:-8}"
runner=(python python/run_solver_characteristics.py)
runner+=(--timeout "${SC_TIMEOUT:-10800}")
runner+=(--threads "${SC_THREADS:-$cpus}")
runner+=(--replicas "${SC_REPLICAS:-5}")
runner+=(--single-thread-replicas "${SC_SINGLE_REPLICAS:-3}")

for family in ${SC_FAMILY:-}; do
  runner+=(--family "$family")
done
for case_name in ${SC_CASE:-}; do
  runner+=(--case "$case_name")
done
if [[ "${SC_ENABLE_PRESCRIBED:-0}" == "1" ]]; then
  runner+=(--enable-prescribed)
fi
for flag in ${SC_PRESCRIBED_FLAGS:-}; do
  runner+=(--prescribed-flag "$flag")
done
# shellcheck disable=SC2206
extra=(${SC_EXTRA_ARGS:-})
runner+=("${extra[@]+"${extra[@]}"}")

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} cpus=${cpus}"
echo "cmd: ${runner[*]}"
./bin/testshell --gtest_filter='test_minimization_report.*:ShellEquilibrium*.*:ZigZagSequenceMetric.*'
status=0
"${runner[@]}" || status=$?
python python/analyze_solver_characteristics.py
python python/visualize_solver_characteristics.py \
  --study-root run/solver_characteristics \
  --out-root run/solver_characteristics/figures \
  --no-cross-mesh --mesh-triplet "" \
  --toolpath-case crown_nested_broad_to_central
exit "$status"
