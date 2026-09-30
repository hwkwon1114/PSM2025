#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=cholmod_integration
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/integration_%j.out
set -eo pipefail
if [[ ${INTEGRATION_WATCHDOG_ACTIVE:-0} != 1 ]]; then
 export INTEGRATION_WATCHDOG_ACTIVE=1
 exec timeout --signal=TERM --kill-after=5 870 bash "$0"
fi
root=/gpfs/projects/p33082/PSM2025-zigzag
out="$root/run/nonconvex_solver_benchmark/integration_${SLURM_JOB_ID}"
mkdir "$out";mkdir "$out/source"
cp "$root/python/qualify_cholmod_integration.py" "$root/python/run_nonconvex_screen.py" "$out/source/"
cp "$root/scripts/cholmod_integration.sbatch" "$out/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$out/protocol_at_launch.md"
export NONCONVEX_WITH_CHOLMOD=1 NONCONVEX_BUILD_SCREEN=1 NONCONVEX_SKIP_PREFLIGHT=1 NONCONVEX_AUDIT_METRIC=0
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""
bash "$root/scripts/benchmark_solver_tests.sbatch"
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
binary="$root/run/nonconvex_solver_benchmark/development_${SLURM_JOB_ID}/source/bin/testshell"
sha256sum "$out/source/"*.py > "$out/source_sha256.txt"
ldd "$binary" > "$out/linked_libraries.txt"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))');echo "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" python "$out/source/qualify_cholmod_integration.py" --root "$out" --binary "$binary" > "$out/qualification.log" 2>&1
printf 'INTEGRATION_QUALIFIED; no scientific fits launched\n' > "$out/complete.txt"
