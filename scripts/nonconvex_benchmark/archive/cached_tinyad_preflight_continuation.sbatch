#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=cached_tinyad_preflight_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:09:20
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/cache_qualification_7516445/continuation_%j.out
set -eo pipefail
if [[ ${CACHE_CONTINUATION_WATCHDOG_ACTIVE:-0} != 1 ]]; then
 export CACHE_CONTINUATION_WATCHDOG_ACTIVE=1
 exec timeout --signal=TERM --kill-after=5 540 bash "$0"
fi
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
prior="$root/run/nonconvex_solver_benchmark/cache_qualification_7516445"
binary="$root/run/nonconvex_solver_benchmark/development_7516445/source/bin/testshell"
out="$prior/continuation_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
mkdir "$out/source"
cp "$prior/runner/run_nonconvex_screen.py" "$out/source/"
cp "$root/python/qualify_cholmod_integration.py" "$root/python/continue_cached_tinyad_preflight.py" "$out/source/"
cp "$root/scripts/cached_tinyad_preflight_continuation.sbatch" "$out/"
find "$out/source" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/source.sha256"
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""
expected=65c76a6482bd36d366b8510e676c18cbcd9d55e9b9d726f351d5994e82888886
actual=$(sha256sum "$binary" | awk '{print $1}')
if [[ "$actual" != "$expected" ]]; then echo 'Qualified build changed; aborting';exit 1;fi
python "$prior/verify_cached_tinyad_source.py" \
 --production "$root/run/nonconvex_solver_benchmark/development_7516445/source/src/libshell/TinyADHessian_Bilayer.hpp" \
 --cached "$root/run/nonconvex_solver_benchmark/development_7516445/source/test/diagnostics/CachedTinyADHessian_Bilayer.hpp" > "$out/built_kernel_identity.json"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))');echo "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" python "$out/source/continue_cached_tinyad_preflight.py" \
 --root "$out" --binary "$binary" --prior "$prior"
