#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=hybrid_transition_smoke
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/hybrid_smoke_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
out="$root/run/nonconvex_solver_benchmark/hybrid_smoke_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
frozen="$root/run/nonconvex_solver_benchmark/development_7546843/source"
mkdir "$out/source"
cp "$root/test/diagnostics/HybridBenchmarkTransition.hpp" "$root/test/diagnostics/SmokeHybridBenchmark.cpp" "$out/source/"
cp "$frozen/test/diagnostics/NonconvexBenchmark.hpp" "$frozen/test/diagnostics/BenchmarkNewtonFactorization.hpp" "$frozen/test/diagnostics/ShiftedNewtonFactorization.hpp" "$out/source/"
sha256sum "$out/source/"* "$frozen/lib/libhlbfgs.a" > "$out/source_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; } > "$out/environment.txt"
python - "$frozen" "$out" <<'PY'
import json,shlex,subprocess,sys
from pathlib import Path
frozen,out=map(Path,sys.argv[1:])
row=next(x for x in json.loads((frozen/'build/compile_commands.json').read_text()) if x['file'].endswith('Test_NonconvexBenchmark.cpp'))
a=shlex.split(row['command']);cmd=[];i=0
while i<len(a):
 if a[i] in ('-o','-c'):i+=2;continue
 if a[i].startswith('-DNONCONVEX_'):i+=1;continue
 cmd.append(a[i]);i+=1
cmd += ['-I'+str(out/'source'),str(out/'source/SmokeHybridBenchmark.cpp'),str(frozen/'lib/libhlbfgs.a'),'-L'+str(Path(sys.executable).parents[1]/'lib'),'-ltbb','-o',str(out/'smoke')]
(out/'compile_command.json').write_text(json.dumps(cmd,indent=2))
subprocess.run(cmd,check=True,timeout=240)
PY
sha256sum "$out/smoke" > "$out/binary.sha256"
"$out/smoke" "$out"
