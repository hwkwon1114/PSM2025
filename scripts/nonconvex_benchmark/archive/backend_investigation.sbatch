#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=cholmod_frozen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:14:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/backend_%j.out
set -eo pipefail
if [[ ${BACKEND_WATCHDOG_ACTIVE:-0} != 1 ]]; then
 export BACKEND_WATCHDOG_ACTIVE=1
 # 46 seconds already charged to failed build 7403440; do not reset budget.
 exec timeout --signal=TERM --kill-after=5 810 bash "$0"
fi
root=/gpfs/projects/p33082/PSM2025-zigzag
out="$root/run/nonconvex_solver_benchmark/backend_${SLURM_JOB_ID}"
ss=/gpfs/software/2025/spack/opt/spack/linux-rhel8-x86_64/gcc-12.4.0/suite-sparse-7.7.0-m42btauyiweee53twzlcgjrlkdfsoffh
mkdir "$out"
cp "$root/scripts/backend_investigation.sbatch" "$root/test/diagnostics/ProfileSparseBackend.cpp" "$root/test/diagnostics/CholmodFrozenFactorization.hpp" "$root/test/diagnostics/ShiftedNewtonFactorization.hpp" "$out/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$out/protocol_at_launch.md"
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; lscpu; "$CXX" --version; for key in PATH LD_LIBRARY_PATH CONDA_PREFIX CXX OMP_NUM_THREADS OMP_THREAD_LIMIT OMP_DYNAMIC OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS CHOLMOD_USE_GPU CUDA_VISIBLE_DEVICES; do printf '%s=%s\n' "$key" "$(printenv "$key" || true)"; done; } > "$out/environment.txt"
sha256sum "$out/"*.cpp "$out/"*.hpp "$out/"*.sbatch > "$out/source_sha256.txt"
find "$CONDA_PREFIX/include/eigen3/Eigen" "$CONDA_PREFIX/include/nlohmann" "$ss/include" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/header_sha256.txt"
ldd "$ss/lib64/libcholmod.so" > "$out/cholmod_dependencies.txt"
if grep -q 'not found' "$out/cholmod_dependencies.txt"; then exit 2; fi
linkdirs=$(awk '/=> \// {sub(/\/[^\/]+$/,"",$3);print $3}' "$out/cholmod_dependencies.txt" | sort -u | paste -sd:)
printf '%s\n' "$linkdirs" > "$out/link_search_paths.txt"
"$CXX" -O3 -DNDEBUG -march=native -std=c++23 -I"$CONDA_PREFIX/include/eigen3" -I"$CONDA_PREFIX/include" -I"$ss/include" \
 "$out/ProfileSparseBackend.cpp" -L"$ss/lib64" -Wl,-rpath,"$ss/lib64" -Wl,-rpath-link,"$linkdirs" -lcholmod -o "$out/profile" > "$out/build.log" 2>&1
sha256sum "$out/profile" > "$out/binary.sha256"
ldd "$out/profile" > "$out/linked_libraries.txt"
if grep -q 'not found' "$out/linked_libraries.txt"; then echo 'Unresolved dependency; no experiment' >&2; exit 2; fi
awk '/=> \// {print $3}' "$out/linked_libraries.txt" | sort -u | xargs sha256sum > "$out/library_sha256.txt"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
echo "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" "$out/profile" > "$out/unit_checks.jsonl" 2> "$out/unit_checks.stderr"
inputs=("$root/run/nonconvex_solver_benchmark/preflight_repair_7294450/preflight/derivatives/initial_trust_region_cg.json"
 "$root/run/nonconvex_solver_benchmark/newton_pilot_7386943/screen/cells/02_cylinder_y_minus_sparse_newton/checkpoint_9.json")
sha256sum "${inputs[@]}" > "$out/input_sha256.txt"
for i in "${!inputs[@]}"; do
 taskset -c "$cpu" "$out/profile" "${inputs[$i]}" > "$out/profile_${i}.jsonl" 2> "$out/profile_${i}.stderr"
done
printf 'BACKEND_PROFILE_COMPLETE; no shell evaluations or optimization fits\n' > "$out/complete.txt"
