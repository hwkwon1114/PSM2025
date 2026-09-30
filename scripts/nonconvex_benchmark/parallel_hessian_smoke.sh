#!/usr/bin/env bash
# Bounded kernel checks only; no solver fits. Run on a compute node:
# SLURM_CONSTRAINT=quest10 srun-here --cpus 4 --mem 8G --time 00:03:00 \
#   --env smcpp_vtk38 bash scripts/nonconvex_benchmark/parallel_hessian_smoke.sh
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
if [[ $# -ne 0 && ( $# -ne 3 || $1 != --parallel-state ) ]]; then
 echo 'usage: parallel_hessian_smoke.sh [--parallel-state bundle state]' >&2; exit 2
fi
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_CPUS_PER_TASK:-0} -ge 4 ]]
: "${CONDA_PREFIX:?Activate smcpp_vtk38}"
frozen="$root/run/nonconvex_solver_benchmark/development_7516445/source"
out="$root/run/nonconvex_solver_benchmark/parallel_hessian_smoke_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=4 OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
# The frozen library was built with native Eigen alignment. Do not silently
# link it to a translation unit targeting a different processor/ABI.
model=$(lscpu | awk -F: '/^Model name:/{gsub(/^[ \t]+/,"",$2);print $2;exit}')
reference=$(awk -F: '/^Model name:/{gsub(/^[ \t]+/,"",$2);print $2;exit}' "$root/run/nonconvex_solver_benchmark/development_7516445/environment.txt")
[[ -n "$model" && "$model" == "$reference" ]]
mkdir "$out/source"
cp "$root/test/diagnostics/LocalDihedralCachedTinyADHessian_Bilayer.hpp" \
   "$root/test/diagnostics/ProfileLocalDihedralCachedHessian.cpp" "$out/source/"
cp "$0" "$out/"
sha256sum "$out/source/"* "$out/parallel_hessian_smoke.sh" "$frozen/lib/"*.a > "$out/source.sha256"
{ hostname; date -Is; lscpu; env | grep -E '^(OMP_|SLURM_|CONDA_PREFIX|MKL_|OPENBLAS_)'; } > "$out/environment.txt"
ss="$root/build_cmake/_deps"
"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" -O3 -DNDEBUG -march=native -mtune=native -std=c++23 -fopenmp \
 -DUSETRILIBRARY -DUSEVTK -DUSEGSL -DUSEHLBFGS '-D__TBB_NO_IMPLICIT_LINKAGE=1' \
 -I"$out/source" -I"$frozen/test/diagnostics" -I"$frozen/src/libshell" -I"$frozen/external/tinyad/include" \
 -I"$ss/libigl-src/include" -I"$ss/triangle-src" \
 -I"$CONDA_PREFIX/include/eigen3" -I"$CONDA_PREFIX/include" -I"$CONDA_PREFIX/include/vtk-9.3" \
 "$out/source/ProfileLocalDihedralCachedHessian.cpp" -o "$out/profile" \
 -Wl,--start-group "$frozen/lib/liblibshell.a" "$frozen/lib/libtriangle.a" "$frozen/lib/libhlbfgs.a" -Wl,--end-group \
 -L"$CONDA_PREFIX/lib" -ltbb -ltbbmalloc -lvtkRenderingCore-9.3 -lvtkFiltersSources-9.3 \
 -lvtkFiltersCore-9.3 -lvtkIOXML-9.3 -lvtkIOXMLParser-9.3 -lvtkCommonExecutionModel-9.3 \
 -lvtkCommonDataModel-9.3 -lvtkCommonTransforms-9.3 -lvtkCommonMisc-9.3 \
 -lvtkCommonMath-9.3 -lvtkCommonCore-9.3 -lvtksys-9.3 -lvtkkissfft-9.3 \
 -lgsl -lgslcblas > "$out/build.log" 2>&1
sha256sum "$out/profile" > "$out/binary.sha256"
ldd "$out/profile" > "$out/runtime.txt"
if [[ $# -eq 0 ]]; then
 set -- --parallel-small-only
else
 sha256sum "$2/"* "$3" > "$out/input.sha256"
fi
printf '%q ' "$@" > "$out/arguments.txt"
timeout --signal=TERM --kill-after=5 60 "$out/profile" "$@" > "$out/checks.jsonl" 2> "$out/checks.stderr"
if [[ -e "$out/input.sha256" ]]; then sha256sum --check "$out/input.sha256" > "$out/input_check.txt"; fi
printf 'PARALLEL_KERNEL_CHECKS_PASSED; no fits\n' | tee "$out/complete.txt"
