#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=local_dihedral_sanitize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:03:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/local_dihedral_sanitize_%j.out
set -eo pipefail
if [[ ${LOCAL_DIHEDRAL_SANITIZE_WATCHDOG:-0} != 1 ]]; then
 export LOCAL_DIHEDRAL_SANITIZE_WATCHDOG=1
 exec timeout --signal=TERM --kill-after=5 150 bash "$0"
fi
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
root=/gpfs/projects/p33082/PSM2025-zigzag
prior="$root/run/nonconvex_solver_benchmark/local_dihedral_smoke_7543030"
frozen="$root/run/nonconvex_solver_benchmark/development_7516445/source"
out="$root/run/nonconvex_solver_benchmark/local_dihedral_sanitize_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
[[ $(<"$prior/exit_code.txt") == 139 && ! -e "$prior/smoke_gate.json" && ! -e "$prior/complete.txt" ]]
sha256sum --check "$prior/source.sha256" "$prior/input.sha256" "$prior/binary.sha256" >/dev/null
mkdir "$out/source"
cp "$prior/source/LocalDihedralCachedTinyADHessian_Bilayer.hpp" "$prior/source/ProfileLocalDihedralCachedHessian.cpp" "$out/source/"
cp "$root/scripts/local_dihedral_sanitize.sbatch" "$out/"
cp "$prior/protocol_at_launch.md" "$out/"
sha256sum "$out/source/"* "$out/"*.sbatch "$out/protocol_at_launch.md" > "$out/source.sha256"
{
 hostname;date -Is;scontrol show job "$SLURM_JOB_ID";lscpu;
 "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" --version;
 for key in PATH LD_LIBRARY_PATH CONDA_PREFIX OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS; do printf '%s=%s\n' "$key" "$(printenv "$key" || true)"; done
} > "$out/environment.txt"
ss="$root/build_cmake/_deps"
"$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" -O1 -g -DNDEBUG -march=nocona -mtune=haswell -std=c++23 \
 -fno-omit-frame-pointer -fsanitize=address,undefined -fopenmp \
 -DUSETRILIBRARY -DUSEVTK -DUSEGSL -DUSEHLBFGS '-D__TBB_NO_IMPLICIT_LINKAGE=1' \
 -I"$out/source" -I"$frozen/test/diagnostics" -I"$frozen/src/libshell" -I"$frozen/external/tinyad/include" \
 -I"$ss/libigl-src/include" -I"$ss/triangle-src" \
 -I"$CONDA_PREFIX/include/eigen3" -I"$CONDA_PREFIX/include" -I"$CONDA_PREFIX/include/vtk-9.3" \
 "$out/source/ProfileLocalDihedralCachedHessian.cpp" -o "$out/profile_asan" \
 -Wl,-rpath,"$CONDA_PREFIX/lib" -Wl,--start-group \
 "$frozen/lib/liblibshell.a" "$frozen/lib/libtriangle.a" "$frozen/lib/libhlbfgs.a" -Wl,--end-group \
 -L"$CONDA_PREFIX/lib" -ltbb -ltbbmalloc -lvtkRenderingCore-9.3 -lvtkFiltersSources-9.3 \
 -lvtkFiltersCore-9.3 -lvtkIOXML-9.3 -lvtkIOXMLParser-9.3 -lvtkCommonExecutionModel-9.3 \
 -lvtkCommonDataModel-9.3 -lvtkCommonTransforms-9.3 -lvtkCommonMisc-9.3 \
 -lvtkCommonMath-9.3 -lvtkCommonCore-9.3 -lvtksys-9.3 -lvtkkissfft-9.3 \
 -lgsl -lgslcblas > "$out/build.log" 2>&1
sha256sum "$out/profile_asan" > "$out/binary.sha256"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
printf '%s\n' "$cpu" > "$out/pinned_cpu.txt"
export ASAN_OPTIONS=abort_on_error=1:detect_leaks=0:print_stacktrace=1:handle_segv=1
export UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1
taskset -c "$cpu" "$out/profile_asan" --small-only > "$out/smoke.jsonl" 2> "$out/smoke.stderr"
echo 'SANITIZED_SMALL_DIAGNOSTIC_RETURNED; not a release smoke gate or full-run authorization' > "$out/complete.txt"
