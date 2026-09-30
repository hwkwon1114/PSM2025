#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=ordering_diagnostics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/ordering_%j.out
set -eo pipefail
# Bound the whole process group, not just Slurm's non-exact time limit.
if [[ ${ORDERING_WATCHDOG_ACTIVE:-0} != 1 ]]; then
 export ORDERING_WATCHDOG_ACTIVE=1
 exec timeout --signal=TERM --kill-after=5 870 bash "$0"
fi
start=$SECONDS
root=/gpfs/projects/p33082/PSM2025-zigzag
out="$root/run/nonconvex_solver_benchmark/ordering_${SLURM_JOB_ID}"
mkdir "$out"
cp "$root/scripts/ordering_investigation.sbatch" "$root/test/diagnostics/ProfileSparseOrdering.cpp" "$out/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$out/protocol_at_launch.md"
export NONCONVEX_BUILD_SCREEN=1 NONCONVEX_SKIP_PREFLIGHT=1 NONCONVEX_AUDIT_METRIC=0
bash "$root/scripts/benchmark_solver_tests.sbatch"
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=false
src="$root/run/nonconvex_solver_benchmark/development_${SLURM_JOB_ID}/source"
cp "$src/test/diagnostics/GeometricNestedDissection.hpp" "$src/test/diagnostics/ShiftedNewtonFactorization.hpp" "$out/"
if (( SECONDS-start > 760 )); then
 echo 'Profiling deferred: insufficient budget after regression build; no retry' > "$out/partial.txt"
 exit 0
fi
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; lscpu; "$CXX" --version; } > "$out/environment.txt"
sha256sum "$out/"*.cpp "$out/"*.hpp "$out/"*.sbatch > "$out/source_sha256.txt"
find "$CONDA_PREFIX/include/eigen3/Eigen" "$CONDA_PREFIX/include/nlohmann" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/dependency_sha256.txt"
"$CXX" -O3 -DNDEBUG -march=native -std=c++23 -I"$CONDA_PREFIX/include/eigen3" -I"$CONDA_PREFIX/include" \
 "$out/ProfileSparseOrdering.cpp" -o "$out/profile" > "$out/profile_build.log" 2>&1
sha256sum "$out/profile" > "$out/binary.sha256"
bundle="$root/run/nonconvex_solver_benchmark/newton_pilot_7386943/preflight/bundle"
inputs=("$root/run/nonconvex_solver_benchmark/preflight_repair_7294450/preflight/derivatives/initial_trust_region_cg.json"
 "$root/run/nonconvex_solver_benchmark/newton_pilot_7386943/screen/cells/02_cylinder_y_minus_sparse_newton/checkpoint_9.json")
sha256sum "$bundle/metadata.json" "$bundle/reference.f64" "$bundle/edges.i32" "${inputs[@]}" > "$out/input_sha256.txt"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
for i in "${!inputs[@]}"; do
 taskset -c "$cpu" "$out/profile" "${inputs[$i]}" "$bundle" > "$out/profile_${i}.jsonl" 2> "$out/profile_${i}.stderr"
done
printf 'ORDERING_PROFILE_COMPLETE; no optimization fits performed\n' > "$out/complete.txt"
