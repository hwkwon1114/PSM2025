#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=cholmod_matched_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/cholmod_pilot_%j.out
set -eo pipefail
if [[ ${PILOT_WATCHDOG_ACTIVE:-0} != 1 ]]; then
    export PILOT_WATCHDOG_ACTIVE=1
    exec timeout --signal=TERM --kill-after=15 1770 bash "$0"
fi

begin=$SECONDS
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u

root=/gpfs/projects/p33082/PSM2025-zigzag
build="$root/run/nonconvex_solver_benchmark/development_7405668"
binary="$build/source/bin/testshell"
out="$root/run/nonconvex_solver_benchmark/cholmod_pilot_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
mkdir "$out/source"

# Verify binary integrity against qualified build
expected_binary_sha="6d9e10bd92efcc2e59fb9c5823eb3bac726c2c7ab8ece695b09d136ec1794a91"
actual_binary_sha=$(sha256sum "$binary" | awk '{print $1}')
if [[ "$actual_binary_sha" != "$expected_binary_sha" ]]; then
    echo "ERROR: binary SHA mismatch! Expected $expected_binary_sha, got $actual_binary_sha" >&2
    exit 1
fi

# Copy runner sources and protocol for strict provenance
cp "$root/python/run_nonconvex_screen.py" "$root/python/run_cholmod_pilot.py" "$out/source/"
cp "$root/scripts/cholmod_matched_pilot.sbatch" "$out/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$out/protocol_at_launch.md"
find "$out/source" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/source_sha256.txt"

# Thread and GPU suppression
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""

{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; lscpu; conda list; } > "$out/environment.txt"
ldd "$binary" > "$out/linked_libraries.txt"

cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
echo "$cpu" > "$out/pinned_cpu.txt"
export PILOT_SETUP_SECONDS=$((SECONDS-begin))
printf 'Pilot root=%s CPU=%s; setup_seconds=%d\n' "$out" "$cpu" "$PILOT_SETUP_SECONDS"

taskset -c "$cpu" python "$out/source/run_cholmod_pilot.py" --root "$out" --binary "$binary"
printf 'CHOLMOD_PILOT_COMPLETE\n' >> "$out/job.log"
