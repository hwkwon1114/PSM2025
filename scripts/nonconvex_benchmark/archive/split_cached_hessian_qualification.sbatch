#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=split_cached_hessian
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/split_cached_hessian_%j.out
set -eo pipefail
# A single bounded attempt; no retry if the build or parity checks fail.
if [[ ${SPLIT_HESSIAN_WATCHDOG_ACTIVE:-0} != 1 ]]; then
 export SPLIT_HESSIAN_WATCHDOG_ACTIVE=1
 exec timeout --signal=TERM --kill-after=5 860 bash "$0"
fi
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
root=/gpfs/projects/p33082/PSM2025-zigzag
frozen="$root/run/nonconvex_solver_benchmark/development_7516445/source"
bundle="$root/run/nonconvex_solver_benchmark/cache_qualification_7516445/continuation_7517700/preflight/bundle"
pilot="$root/run/nonconvex_solver_benchmark/cholmod_pilot_7494613"
prior="$root/run/nonconvex_solver_benchmark/cache_curved_timing_7518799/continuation_7519129"
plus="$pilot/screen/cells/01_cylinder_y_plus_sparse_newton/selected_x.f64"
minus="$pilot/screen/cells/02_cylinder_y_minus_sparse_newton/selected_x.f64"
out="$root/run/nonconvex_solver_benchmark/split_cached_hessian_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
mkdir -p "$out/source" "$out/input"
cp "$root/test/diagnostics/SplitCachedTinyADHessian_Bilayer.hpp" \
   "$root/test/diagnostics/ProfileSplitCachedHessian.cpp" \
   "$root/python/analyze_split_cached_hessian.py" "$out/source/"
cp "$root/scripts/split_cached_hessian_qualification.sbatch" "$out/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$out/protocol_at_launch.md"
sha256sum "$out/source/"* "$out/"*.sbatch "$out/protocol_at_launch.md" > "$out/source.sha256"
cmp "$bundle/manifest.json" "$pilot/preflight/bundle/manifest.json"
printf '%s  %s\n' 7591e5c69abe16d076cf913806673a1b7f50672c08c11c0a81a579787ab17281 "$bundle/manifest.json" \
 4eb0bd8d66ce217e973b77d6529380603b055061eea5f50769f6585fdfaf3aca "$bundle/cylinder_y_plus.f64" \
 7032c08cdb2ff9bcd631b6b0311276433a473a485be4fcf2550eb1e8664bc2e1 "$plus" \
 d7ae525b31bf5b19f43ed7898cb3f130dcc72747780cd38a70a6023d17e31661 "$minus" | sha256sum --check
printf '%s  %s\n' a9b362767ec746e1f03175ed43442214877bdf30b12fa0cec0071f6f7bf1e182 "$frozen/src/libshell/TinyADHessian_Bilayer.hpp" \
 550296be425df6b093239db0b9e990a4fae399e12b85860b30693d62444b7ac1 "$frozen/test/diagnostics/CachedTinyADHessian_Bilayer.hpp" | sha256sum --check
sha256sum --check "$prior/binary.sha256"
mkdir -p "$out/input/bundle"
cp -a "$bundle/." "$out/input/bundle/"
cp "$plus" "$out/input/plus_capped.f64"
cp "$minus" "$out/input/minus_capped.f64"
sha256sum "$out/input/bundle/"* "$out/input/"*.f64 > "$out/input.sha256"
{
 hostname;date -Is;scontrol show job "$SLURM_JOB_ID";lscpu;
 "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" --version;
 for key in PATH LD_LIBRARY_PATH CONDA_PREFIX OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS; do printf '%s=%s\n' "$key" "$(printenv "$key" || true)"; done
} > "$out/environment.txt"
ss="$root/build_cmake/_deps"
"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" -O3 -DNDEBUG -march=native -mtune=native -std=c++23 -fopenmp \
 -DUSETRILIBRARY -DUSEVTK -DUSEGSL -DUSEHLBFGS '-D__TBB_NO_IMPLICIT_LINKAGE=1' \
 -I"$out/source" -I"$frozen/test/diagnostics" -I"$frozen/src/libshell" -I"$frozen/external/tinyad/include" \
 -I"$ss/libigl-src/include" -I"$ss/triangle-src" \
 -I"$CONDA_PREFIX/include/eigen3" -I"$CONDA_PREFIX/include" -I"$CONDA_PREFIX/include/vtk-9.3" \
 "$out/source/ProfileSplitCachedHessian.cpp" -o "$out/profile" \
 -Wl,--start-group "$frozen/lib/liblibshell.a" "$frozen/lib/libtriangle.a" "$frozen/lib/libhlbfgs.a" -Wl,--end-group \
 -L"$CONDA_PREFIX/lib" -ltbb -ltbbmalloc -lvtkRenderingCore-9.3 -lvtkFiltersSources-9.3 \
 -lvtkFiltersCore-9.3 -lvtkIOXML-9.3 -lvtkIOXMLParser-9.3 -lvtkCommonExecutionModel-9.3 \
 -lvtkCommonDataModel-9.3 -lvtkCommonTransforms-9.3 -lvtkCommonMisc-9.3 \
 -lvtkCommonMath-9.3 -lvtkCommonCore-9.3 -lvtksys-9.3 -lvtkkissfft-9.3 \
 -lgsl -lgslcblas > "$out/build.log" 2>&1
sha256sum "$out/profile" > "$out/binary.sha256"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
printf '%s\n' "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" "$out/profile" "$out/input/bundle" "$out/input/plus_capped.f64" \
 "$out/input/minus_capped.f64" "$out/input/bundle/cylinder_y_plus.f64" > "$out/profile.jsonl" 2> "$out/profile.stderr"
python "$out/source/analyze_split_cached_hessian.py" "$out" > "$out/analysis.log"
echo 'SPLIT_STRETCH_CHECKS_AND_TIMING_COMPLETE; no optimizer steps or scientific fits' > "$out/complete.txt"
