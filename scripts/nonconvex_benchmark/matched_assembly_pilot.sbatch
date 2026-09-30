#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=matched_assembly_pilot
#SBATCH --time=00:32:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/matched_assembly_pilot_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
: "${MATCHED_PILOT_DRYRUN_JOB:?passing no-fit ledger dry-run required}"
prior="$root/run/nonconvex_solver_benchmark/matched_pilot_dryrun_${MATCHED_PILOT_DRYRUN_JOB}"
test "$(<"$prior/exit_code.txt")" = 0
test -f "$prior/dry_run_gate.json"
out="$root/run/nonconvex_solver_benchmark/matched_assembly_pilot_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_DYNAMIC=false OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""
# Reuse the exact driver that passed the no-fit ledger dry-run; never copy a
# later working-tree edit or mutate the qualification-only gate.
cp "$prior/run_hessian_assembly_matched_pilot.py" "$prior/qualify_cholmod_integration.py" "$out/"
cp "$root/docs/nonconvex_solver_benchmark_plan.md" "$out/protocol_at_launch.md"
sha256sum "$out"/*.py "$root/run/nonconvex_solver_benchmark/development_7546843/source/bin/testshell" > "$out/source_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; } > "$out/environment.txt"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
echo "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" python "$out/run_hessian_assembly_matched_pilot.py" pilot \
  --root "$out" \
  --development "$root/run/nonconvex_solver_benchmark/development_7546843" \
  --qualification "$root/run/nonconvex_solver_benchmark/local_dihedral_solver_gate_7547943" \
  --smoke "$root/run/nonconvex_solver_benchmark/matched_pilot_smoke_7550736" \
  --dry-run-gate "$prior/dry_run_gate.json"
