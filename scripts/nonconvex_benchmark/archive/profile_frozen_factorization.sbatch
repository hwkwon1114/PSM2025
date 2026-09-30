#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=frozen_factor_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:04:00
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/factor_profile_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
out="$root/run/nonconvex_solver_benchmark/factor_profile_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/build_and_profile.log") 2>&1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=false
cp "$root/test/diagnostics/ProfileFrozenFactorization.cpp" "$root/test/diagnostics/ShiftedNewtonFactorization.hpp" "$root/scripts/profile_frozen_factorization.sbatch" "$out/"
input="$root/run/nonconvex_solver_benchmark/preflight_repair_7294450/preflight/derivatives/initial_trust_region_cg.json"
sha256sum "$input" "$out/ProfileFrozenFactorization.cpp" "$out/ShiftedNewtonFactorization.hpp" "$out/profile_frozen_factorization.sbatch" > "$out/sha256.txt"
find "$CONDA_PREFIX/include/eigen3/Eigen" "$CONDA_PREFIX/include/nlohmann" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/dependency_sha256.txt"
{ hostname; date -Is; lscpu; scontrol show job "$SLURM_JOB_ID"; "$CXX" --version; } > "$out/environment.txt"
"$CXX" -O3 -DNDEBUG -march=native -std=c++23 -I"$CONDA_PREFIX/include/eigen3" -I"$CONDA_PREFIX/include" \
 "$out/ProfileFrozenFactorization.cpp" -o "$out/profile"
sha256sum "$out/profile" >> "$out/sha256.txt"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
if [[ ${EARLY_CHOLESKY:-0} == 1 ]]; then
 # Predeclared: initial state, earliest and latest saved cached models in the
 # first TR cell. These are not independent problems or energy-selected states.
 inputs=("$input"
  "$root/run/nonconvex_solver_benchmark/screen_7296605/cells/03_cylinder_y_plus_trust_region_cg/checkpoint_1.json"
  "$root/run/nonconvex_solver_benchmark/screen_7296605/cells/03_cylinder_y_plus_trust_region_cg/checkpoint_3.json")
 for i in "${!inputs[@]}"; do
  sha256sum "${inputs[$i]}" >> "$out/sha256.txt"
  taskset -c "$cpu" "$out/profile" "${inputs[$i]}" --early-cholesky > "$out/profile_${i}.jsonl"
 done
else
 taskset -c "$cpu" "$out/profile" "$input" > "$out/profile.jsonl"
fi
# Inspect the actual initial TR decisions, not just a reconstructed first CG step.
python - "$root" "$out" <<'PY'
import json,sys,hashlib
from pathlib import Path
root,out=map(Path,sys.argv[1:])
ledger_path=root/'run/nonconvex_solver_benchmark/screen_7296605/ledger.json'
ledger=json.loads(ledger_path.read_text());cell=next(c for c in ledger['cells'] if c['index']==3)
p=Path(cell['result']['checkpoint']);raw=p.read_bytes();j=json.loads(raw)
rows=j['payload']['state']['trace'][:5]
with (out/'initial_tr_decisions.json').open('x') as f:
 json.dump(dict(scope='first five attempts of cylinder_y_plus only',checkpoint=str(p),sha256=hashlib.sha256(raw).hexdigest(),trace=rows),f,indent=2)
PY
printf 'FROZEN_PROFILE_COMPLETE; no optimizer was advanced\n'
