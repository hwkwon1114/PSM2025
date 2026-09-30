#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=cache_curved_compile_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:07:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/cache_curved_timing_7518799/continuation_%j.out
set -eo pipefail
if [[ ${CACHE_TIMING_CONTINUATION_WATCHDOG_ACTIVE:-0} != 1 ]]; then
 export CACHE_TIMING_CONTINUATION_WATCHDOG_ACTIVE=1
 exec timeout --signal=TERM --kill-after=5 390 bash "$0"
fi
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
root=/gpfs/projects/p33082/PSM2025-zigzag
frozen="$root/run/nonconvex_solver_benchmark/development_7516445/source"
prior="$root/run/nonconvex_solver_benchmark/cache_curved_timing_7518799"
out="$prior/continuation_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
cp "$root/scripts/cached_curved_timing_continuation.sbatch" "$out/"
cp "$prior/protocol_at_launch.md" "$out/"
sha256sum "$out/"*.sbatch "$out/protocol_at_launch.md" "$prior/source/ProfileCachedHessian.cpp" "$prior/source/analyze_cached_curved_timing.py" > "$out/source.sha256"
[[ $(<"$prior/exit_code.txt") == 1 ]]
[[ ! -e "$prior/profile" && ! -e "$prior/analysis.json" && ! -e "$prior/complete.txt" ]]
grep -q 'fatal error: CachedTinyADHessian_Bilayer.hpp: No such file or directory' "$prior/build.log"
cmp "$root/test/diagnostics/ProfileCachedHessian.cpp" "$prior/source/ProfileCachedHessian.cpp"
sha256sum --check "$prior/input.sha256" >/dev/null
printf '%s  %s\n' a9b362767ec746e1f03175ed43442214877bdf30b12fa0cec0071f6f7bf1e182 "$frozen/src/libshell/TinyADHessian_Bilayer.hpp" 550296be425df6b093239db0b9e990a4fae399e12b85860b30693d62444b7ac1 "$frozen/test/diagnostics/CachedTinyADHessian_Bilayer.hpp" | sha256sum --check
{
 hostname;date -Is;scontrol show job "$SLURM_JOB_ID";lscpu;
 "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" --version;
 for key in PATH LD_LIBRARY_PATH CONDA_PREFIX OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS; do printf '%s=%s\n' "$key" "$(printenv "$key" || true)"; done
} > "$out/environment.txt"
ss="$root/build_cmake/_deps"
"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" -O3 -DNDEBUG -march=native -mtune=native -std=c++23 -fopenmp \
 -DUSETRILIBRARY -DUSEVTK -DUSEGSL -DUSEHLBFGS '-D__TBB_NO_IMPLICIT_LINKAGE=1' \
 -I"$prior/source" -I"$frozen/test/diagnostics" -I"$frozen/src/libshell" -I"$frozen/external/tinyad/include" \
 -I"$ss/libigl-src/include" -I"$ss/triangle-src" \
 -I"$CONDA_PREFIX/include/eigen3" -I"$CONDA_PREFIX/include" -I"$CONDA_PREFIX/include/vtk-9.3" \
 "$prior/source/ProfileCachedHessian.cpp" -o "$out/profile" \
 -Wl,--start-group "$frozen/lib/liblibshell.a" "$frozen/lib/libtriangle.a" "$frozen/lib/libhlbfgs.a" -Wl,--end-group \
 -L"$CONDA_PREFIX/lib" -ltbb -ltbbmalloc -lvtkRenderingCore-9.3 -lvtkFiltersSources-9.3 \
 -lvtkFiltersCore-9.3 -lvtkIOXML-9.3 -lvtkIOXMLParser-9.3 -lvtkCommonExecutionModel-9.3 \
 -lvtkCommonDataModel-9.3 -lvtkCommonTransforms-9.3 -lvtkCommonMisc-9.3 \
 -lvtkCommonMath-9.3 -lvtkCommonCore-9.3 -lvtksys-9.3 -lvtkkissfft-9.3 \
 -lgsl -lgslcblas > "$out/build.log" 2>&1
sha256sum "$out/profile" > "$out/binary.sha256"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
printf '%s\n' "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" "$out/profile" "$prior/input/bundle" "$prior/input/plus_capped.f64" \
 "$prior/input/minus_capped.f64" "$prior/input/bundle/cylinder_y_plus.f64" > "$out/profile.jsonl" 2> "$out/profile.stderr"
python "$prior/source/analyze_cached_curved_timing.py" "$out" > "$out/analysis.log"
echo 'MATCHED_FROZEN_KERNEL_TIMING_COMPLETE; no optimizer steps or scientific fits' > "$out/complete.txt"
