#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=local_dihedral_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:09:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/local_dihedral_full_%j.out
set -eo pipefail
if [[ ${LOCAL_DIHEDRAL_FULL_WATCHDOG:-0} != 1 ]]; then
 export LOCAL_DIHEDRAL_FULL_WATCHDOG=1
 exec timeout --signal=TERM --kill-after=5 510 bash "$0"
fi
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
: "${SMOKE_JOB_ID:?pass the confirmed completed smoke job ID with sbatch --export}"
[[ "$SMOKE_JOB_ID" =~ ^[0-9]+$ ]]
root=/gpfs/projects/p33082/PSM2025-zigzag
smoke="$root/run/nonconvex_solver_benchmark/local_dihedral_smoke_${SMOKE_JOB_ID}"
out="$root/run/nonconvex_solver_benchmark/local_dihedral_full_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
cp "$root/scripts/local_dihedral_full.sbatch" "$out/"
cp "$smoke/protocol_at_launch.md" "$out/"
printf 'verified-smoke-root=%s\n' "$smoke" > "$out/smoke_source.txt"
[[ $(<"$smoke/exit_code.txt") == 0 && -s "$smoke/complete.txt" && ! -e "$smoke/analysis.json" ]]
smoke_model=$(awk -F: '/^Model name:/{gsub(/^[ \t]+/,"",$2);print $2;exit}' "$smoke/environment.txt")
current_model=$(lscpu | awk -F: '/^Model name:/{gsub(/^[ \t]+/,"",$2);print $2;exit}')
[[ -n "$smoke_model" && "$smoke_model" == "$current_model" ]] || { echo 'CPU model differs from native-compiled smoke'; exit 1; }
sha256sum --check "$smoke/source.sha256" "$smoke/input.sha256" "$smoke/binary.sha256" >/dev/null
cp "$smoke/profile" "$out/profile"
python - "$smoke" "$out/profile" <<'PY'
import hashlib,json,sys
from pathlib import Path
s=Path(sys.argv[1]);binary=Path(sys.argv[2]);gate=json.loads((s/'smoke_gate.json').read_text())
assert gate['passed'] and gate['smoke_only'] and gate['scientific_fits']==0
assert gate['binary_sha256']==hashlib.sha256(binary.read_bytes()).hexdigest()
for name,key in [('LocalDihedralCachedTinyADHessian_Bilayer.hpp','candidate_header_sha256'),
                 ('ProfileLocalDihedralCachedHessian.cpp','profiler_sha256')]:
 assert hashlib.sha256((s/'source'/name).read_bytes()).hexdigest()==gate[key]
PY
PYTHONPATH="$smoke/source" python - "$smoke" "$out" <<'PY'
import json,sys
from pathlib import Path
from qualify_cholmod_integration import runtime,digest
smoke,out=map(Path,sys.argv[1:]);prior=json.loads((smoke/'runtime_manifest.json').read_text())
current=runtime(out/'profile')
assert current['binary_sha256']==prior['binary_sha256']
assert current['libraries']==prior['libraries'],'resolved runtime libraries changed across nodes'
assert current['loader_environment']==prior['loader_environment'],'loader environment changed'
current['identity']=digest(current)
(out/'runtime_manifest.json').write_text(json.dumps(current,indent=2)+'\n')
PY
{
 hostname;date -Is;scontrol show job "$SLURM_JOB_ID";lscpu;
 for key in PATH LD_LIBRARY_PATH CONDA_PREFIX OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS BLIS_NUM_THREADS; do printf '%s=%s\n' "$key" "$(printenv "$key" || true)"; done
} > "$out/environment.txt"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
printf '%s\n' "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" "$out/profile" --full-only "$smoke/input/bundle" "$smoke/input/plus_capped.f64" \
 "$smoke/input/minus_capped.f64" "$smoke/input/bundle/cylinder_y_plus.f64" > "$out/profile.jsonl" 2> "$out/profile.stderr"
PYTHONPATH="$smoke/source" python - "$out" <<'PY'
import json,sys
from pathlib import Path
from qualify_cholmod_integration import runtime,verify,digest
out=Path(sys.argv[1]);manifest=json.loads((out/'runtime_manifest.json').read_text())
verify(out/'profile',manifest)
assert digest(runtime(out/'profile'))==manifest['identity'],'resolved runtime changed during full check'
PY
sha256sum --check "$smoke/input.sha256" >/dev/null
python "$smoke/source/analyze_local_dihedral_cached_hessian.py" "$out" "$smoke" > "$out/analysis.log"
echo 'FULL_SIZE_PARITY_AND_TIMING_COMPLETE; no optimizer steps or scientific fits' > "$out/complete.txt"
