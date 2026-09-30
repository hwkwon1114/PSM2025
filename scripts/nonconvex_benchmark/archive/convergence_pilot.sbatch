#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=convergence_pilot
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/convergence_pilot_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
: "${CONVERGENCE_PREP_JOB:?passing preparation job required}"
prep="$root/run/nonconvex_solver_benchmark/convergence_prepare_${CONVERGENCE_PREP_JOB}"
test "$(<"$prep/exit_code.txt")" = 0
test -f "$prep/gate.json"
out="$root/run/nonconvex_solver_benchmark/convergence_pilot_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=false CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""
cp "$prep/run_convergence_benchmark.py" "$prep/qualify_cholmod_integration.py" "$out/"
cp "$root/docs/nonconvex_solver_benchmark_plan.md" "$out/plan_at_launch.md"
sha256sum "$out/"*.py "$prep/convergence_worker" > "$out/source_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; } > "$out/environment.txt"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
echo "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" python "$out/run_convergence_benchmark.py" pilot --out "$out" --frozen "$root/run/nonconvex_solver_benchmark/development_7546843/source" --preparation "$prep"
