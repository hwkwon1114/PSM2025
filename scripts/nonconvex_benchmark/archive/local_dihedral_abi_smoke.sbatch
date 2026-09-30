#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=local_dihedral_abi_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/local_dihedral_abi_smoke_%j.out
set -eo pipefail
if [[ ${LOCAL_DIHEDRAL_ABI_SMOKE_WATCHDOG:-0} != 1 ]]; then
 export LOCAL_DIHEDRAL_ABI_SMOKE_WATCHDOG=1
 exec timeout --signal=TERM --kill-after=5 95 bash "$0"
fi
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
root=/gpfs/projects/p33082/PSM2025-zigzag
frozen="$root/run/nonconvex_solver_benchmark/development_7516445/source"
prior="$root/run/nonconvex_solver_benchmark/local_dihedral_smoke_7543030"
bundle="$root/run/nonconvex_solver_benchmark/cache_qualification_7516445/continuation_7517700/preflight/bundle"
pilot="$root/run/nonconvex_solver_benchmark/cholmod_pilot_7494613"
plus="$pilot/screen/cells/01_cylinder_y_plus_sparse_newton/selected_x.f64"
minus="$pilot/screen/cells/02_cylinder_y_minus_sparse_newton/selected_x.f64"
out="$root/run/nonconvex_solver_benchmark/local_dihedral_abi_smoke_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
[[ $(<"$prior/exit_code.txt") == 139 && ! -e "$prior/smoke_gate.json" ]]
sha256sum --check "$prior/source.sha256" "$prior/input.sha256" "$prior/binary.sha256" >/dev/null
mkdir -p "$out/source" "$out/input"
cp "$prior/source/LocalDihedralCachedTinyADHessian_Bilayer.hpp" \
 "$prior/source/ProfileLocalDihedralCachedHessian.cpp" \
 "$prior/source/analyze_local_dihedral_cached_hessian.py" \
 "$prior/source/verify_local_dihedral_smoke.py" \
 "$prior/source/qualify_cholmod_integration.py" \
 "$prior/source/run_nonconvex_screen.py" "$out/source/"
cp "$root/scripts/local_dihedral_abi_smoke.sbatch" "$out/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$out/protocol_at_launch.md"
sha256sum "$out/source/"* "$out/"*.sbatch "$out/protocol_at_launch.md" > "$out/source.sha256"
printf '%s  %s\n' a9b362767ec746e1f03175ed43442214877bdf30b12fa0cec0071f6f7bf1e182 "$frozen/src/libshell/TinyADHessian_Bilayer.hpp" \
 550296be425df6b093239db0b9e990a4fae399e12b85860b30693d62444b7ac1 "$frozen/test/diagnostics/CachedTinyADHessian_Bilayer.hpp" | sha256sum --check
cmp "$bundle/manifest.json" "$pilot/preflight/bundle/manifest.json"
printf '%s  %s\n' 7591e5c69abe16d076cf913806673a1b7f50672c08c11c0a81a579787ab17281 "$bundle/manifest.json" \
 4eb0bd8d66ce217e973b77d6529380603b055061eea5f50769f6585fdfaf3aca "$bundle/cylinder_y_plus.f64" \
 7032c08cdb2ff9bcd631b6b0311276433a473a485be4fcf2550eb1e8664bc2e1 "$plus" \
 d7ae525b31bf5b19f43ed7898cb3f130dcc72747780cd38a70a6023d17e31661 "$minus" | sha256sum --check
python - "$bundle" "$out/source/run_nonconvex_screen.py" <<'PY'
import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('frozen_runner',sys.argv[2]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert m.verify_bundle(Path(sys.argv[1]))=='7591e5c69abe16d076cf913806673a1b7f50672c08c11c0a81a579787ab17281'
PY
cp -a "$bundle" "$out/input/bundle"
cp "$plus" "$out/input/plus_capped.f64"
cp "$minus" "$out/input/minus_capped.f64"
sha256sum "$out/input/bundle/"* "$out/input/"*.f64 > "$out/input.sha256"
{
 hostname;date -Is;scontrol show job "$SLURM_JOB_ID";lscpu;
 "$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" --version;
 for key in PATH LD_LIBRARY_PATH CONDA_PREFIX OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS; do printf '%s=%s\n' "$key" "$(printenv "$key" || true)"; done
} > "$out/environment.txt"
model=$(lscpu | awk -F: '/^Model name:/{gsub(/^[ \t]+/,"",$2);print $2;exit}')
reference_model=$(awk -F: '/^Model name:/{gsub(/^[ \t]+/,"",$2);print $2;exit}' "$root/run/nonconvex_solver_benchmark/development_7516445/environment.txt")
[[ -n "$model" && "$model" == "$reference_model" ]] || { echo 'CPU model mismatch with frozen native library';exit 1; }
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
PYTHONPATH="$out/source" python - "$out" <<'PY'
import json,sys
from pathlib import Path
from qualify_cholmod_integration import runtime,digest
out=Path(sys.argv[1]);state=runtime(out/'profile');state['identity']=digest(state)
(out/'runtime_manifest.json').write_text(json.dumps(state,indent=2)+'\n')
PY
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
printf '%s\n' "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" "$out/profile" --small-only > "$out/smoke.jsonl" 2> "$out/smoke.stderr"
PYTHONPATH="$out/source" python - "$out" <<'PY'
import json,sys
from pathlib import Path
from qualify_cholmod_integration import runtime,verify,digest
out=Path(sys.argv[1]);manifest=json.loads((out/'runtime_manifest.json').read_text())
verify(out/'profile',manifest)
assert digest(runtime(out/'profile'))==manifest['identity'],'resolved runtime changed during smoke'
PY
sha256sum --check "$out/input.sha256" "$out/binary.sha256" >/dev/null
python "$out/source/verify_local_dihedral_smoke.py" "$out" > "$out/smoke.log"
echo 'SMALL_ONLY_SMOKE_COMPLETE; full-size run awaits researcher confirmation' > "$out/complete.txt"
