#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=local_dihedral_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/local_dihedral_trace_%j.out
set -eo pipefail
# Single fault-localizing invocation; no build, full-size state or scientific fit.
if [[ ${LOCAL_DIHEDRAL_TRACE_WATCHDOG:-0} != 1 ]]; then
 export LOCAL_DIHEDRAL_TRACE_WATCHDOG=1
 exec timeout --signal=TERM --kill-after=5 95 bash "$0"
fi
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
root=/gpfs/projects/p33082/PSM2025-zigzag
prior="$root/run/nonconvex_solver_benchmark/local_dihedral_smoke_7543030"
out="$root/run/nonconvex_solver_benchmark/local_dihedral_trace_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
cp "$root/scripts/local_dihedral_stacktrace.sbatch" "$out/"
cp "$prior/protocol_at_launch.md" "$out/"
sha256sum "$out/"*.sbatch "$out/protocol_at_launch.md" > "$out/source.sha256"
[[ $(<"$prior/exit_code.txt") == 139 && ! -e "$prior/smoke_gate.json" ]]
sha256sum --check "$prior/binary.sha256" "$prior/source.sha256" "$prior/input.sha256" >/dev/null
[[ -r /lib64/libSegFault.so ]]
sha256sum /lib64/libSegFault.so "$prior/profile" > "$out/runtime_additions.sha256"
{
 hostname;date -Is;scontrol show job "$SLURM_JOB_ID";lscpu;
 for key in PATH LD_LIBRARY_PATH CONDA_PREFIX OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS; do printf '%s=%s\n' "$key" "$(printenv "$key" || true)"; done
} > "$out/environment.txt"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
printf '%s\n' "$cpu" > "$out/pinned_cpu.txt"
ulimit -c 0
export LD_PRELOAD=/lib64/libSegFault.so
export SEGFAULT_USE_ALTSTACK=1
export SEGFAULT_SIGNALS=all
set +e
taskset -c "$cpu" "$prior/profile" --small-only > "$out/small.jsonl" 2> "$out/small.stderr"
rc=$?
set -e
printf 'small-only diagnostic process exit=%s; this invocation cannot qualify the original smoke or authorize full size\n' "$rc" > "$out/diagnostic_status.txt"
exit "$rc"
