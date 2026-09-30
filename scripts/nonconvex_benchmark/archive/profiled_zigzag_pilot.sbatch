#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=profiled_hybrid_pilot
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/profiled_zigzag_pilot_%j.out
set -eo pipefail
export PROFILED_PHASE_ENTRY=$(date +%s)
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
out="$root/run/nonconvex_solver_benchmark/profiled_zigzag_pilot_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=false CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
for name in run_profiled_zigzag_pilot report_profiled_zigzag_pilot audit_convergence_benchmark run_convergence_benchmark run_nonconvex_screen qualify_cholmod_integration; do
 cp "$root/python/${name}.py" "$out/"
done
cp "$root/run/zigzag_abc_preview/optimization_request.md" "$out/request_at_launch.md"
cp "$root/run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/gate.json" "$out/target_gate_at_launch.json"
cp "$root/run/nonconvex_solver_benchmark/convergence_prepare_7592200/gate.json" "$out/worker_gate_at_launch.json"
cp "$root/docs/nonconvex_solver_benchmark_plan.md" "$out/plan_at_launch.md"
sha256sum "$out/"*.py "$root/run/nonconvex_solver_benchmark/convergence_prepare_7592200/convergence_worker" > "$out/source_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; } > "$out/environment.txt"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
echo "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" python "$out/run_profiled_zigzag_pilot.py" --out "$out" --root "$root"
