#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=switch_sweep
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/switch_sweep_%j.out
set -eo pipefail
export PROFILED_PHASE_ENTRY=$(date +%s)
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
qualified="$root/run/nonconvex_solver_benchmark/parallel_solver_gate_7723520"
out="$root/run/nonconvex_solver_benchmark/switch_sweep_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT

# 4 OpenMP threads for Hessian evaluation, 1 thread for serial BLAS/L-BFGS
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=4 OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg

for name in run_lbfgs_switch_sweep report_lbfgs_switch_sweep run_convergence_benchmark run_nonconvex_screen qualify_cholmod_integration; do
 cp "$root/python/${name}.py" "$out/"
done
cp "$0" "$out/lbfgs_switch_sweep.sbatch"
cp "$qualified/gate.json" "$out/worker_gate_at_launch.json"
cp "$qualified/runtime_manifest.json" "$out/qualified_runtime_at_launch.json"
cp "$root/run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/gate.json" "$out/target_gate_at_launch.json"
cp "$root/docs/nonconvex_solver_benchmark_plan.md" "$out/plan_at_launch.md"

sha256sum "$out/"*.py "$out/"*.sbatch "$qualified/convergence_worker" > "$out/source_sha256.txt"
sha256sum "$root/src/libshell/TinyADHessian_Bilayer.hpp" "$root/src/libshell/ShellEquilibriumSolver.hpp" "$root/external/hlbfgs/HLBFGS.cpp" > "$out/production_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; lscpu; env | grep -E '^(OMP_|MKL_|OPENBLAS_|BLIS_|LD_)'; } > "$out/environment.txt"

cpus=$(python -c 'import os; a=sorted(os.sched_getaffinity(0)); assert len(a)>=4; print(",".join(map(str,a[:4])))')
echo "$cpus" > "$out/pinned_cpus.txt"

echo "Executing L-BFGS Switch Sweep on 4 CPUs..."
taskset -c "$cpus" python "$out/run_lbfgs_switch_sweep.py" --out "$out" --root "$root" --gate "$qualified"

echo "Generating publication figures and summary..."
python "$out/report_lbfgs_switch_sweep.py" --root "$out"

sha256sum -c "$out/source_sha256.txt" "$out/production_sha256.txt" > "$out/final_hash_check.txt"
echo "L_BFGS_SWITCH_SWEEP_COMPLETE"
