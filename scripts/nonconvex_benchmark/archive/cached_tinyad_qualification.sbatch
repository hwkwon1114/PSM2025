#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=cached_tinyad_qualification
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/cache_qualification_%j.out
set -eo pipefail
# One authorization, one attempt; terminate before scheduler cap, no scientific fits.
if [[ ${CACHE_WATCHDOG_ACTIVE:-0} != 1 ]]; then
 export CACHE_WATCHDOG_ACTIVE=1
 exec timeout --signal=TERM --kill-after=5 1175 bash "$0"
fi
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
out="$root/run/nonconvex_solver_benchmark/cache_qualification_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
cp "$root/scripts/cached_tinyad_qualification.sbatch" "$out/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$out/plan_at_launch.md"
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export NONCONVEX_TEST_CACHED_TINYAD=1 NONCONVEX_BUILD_SCREEN=1 NONCONVEX_SKIP_PREFLIGHT=1 NONCONVEX_WITH_CHOLMOD=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""
python "$root/python/verify_cached_tinyad_source.py" \
 --production "$root/src/libshell/TinyADHessian_Bilayer.hpp" \
 --cached "$root/test/diagnostics/CachedTinyADHessian_Bilayer.hpp" > "$out/kernel_identity.json"
# Isolated snapshot/compilation, regressions (including small free/clamped parity).
bash "$root/scripts/benchmark_solver_tests.sbatch"
build="$root/run/nonconvex_solver_benchmark/development_${SLURM_JOB_ID}"
cp "$root/python/verify_cached_tinyad_source.py" "$out/"
python "$out/verify_cached_tinyad_source.py" \
 --production "$build/source/src/libshell/TinyADHessian_Bilayer.hpp" \
 --cached "$build/source/test/diagnostics/CachedTinyADHessian_Bilayer.hpp" > "$out/built_kernel_identity.json"
sha256sum "$build/source/bin/testshell" > "$out/worker.sha256"
mkdir "$out/runner";cp "$root/python/run_nonconvex_screen.py" "$out/runner/"
sha256sum "$out/runner/run_nonconvex_screen.py" "$out/verify_cached_tinyad_source.py" "$root/test/diagnostics/CachedTinyADHessian_Bilayer.hpp" > "$out/source.sha256"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))');echo "$cpu" > "$out/pinned_cpu.txt"
# Read-only parity comparison of the same physical Hessians on all six starts;
# includes toy safe-pause/restart and fake ledger tests, but no real fit.
taskset -c "$cpu" python "$out/runner/run_nonconvex_screen.py" preflight \
 --root "$out/preflight" --binary "$build/source/bin/testshell" \
 --newton-backend cholmod --cache-parity
python - "$out/preflight/gate.json" <<'PY'
import json,sys
from pathlib import Path
j=json.loads(Path(sys.argv[1]).read_text())
assert j['passed'] and j['cache_parity'] and j['qualification_only'] and j['four_cell_pilot_ledger_tests_passed']
print('CACHE_PARITY_QUALIFIED_ONLY; zero scientific fits')
PY
