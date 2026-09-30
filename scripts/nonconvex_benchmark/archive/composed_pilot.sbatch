#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=composed_pilot
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --constraint=quest10
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/composed_pilot_%j.out
set -eo pipefail
export PROFILED_PHASE_ENTRY=$(date +%s)
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
qualified="$root/run/nonconvex_solver_benchmark/composed_solver_gate_7721857"
export COMPOSED_QUALIFIED_GATE="$qualified"
out="$root/run/nonconvex_solver_benchmark/composed_pilot_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT
# Four OpenMP threads allocated for parallel Hessian assembly.
# Coordinate-stable symbolic factorization reuse enabled.
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=4 OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
for name in run_profiled_zigzag_pilot report_profiled_zigzag_pilot report_guarded_hessian_pilot audit_convergence_benchmark run_convergence_benchmark run_nonconvex_screen qualify_cholmod_integration; do
 cp "$root/python/${name}.py" "$out/"
done
cp "$0" "$out/composed_pilot.sbatch"
cp "$qualified/gate.json" "$out/worker_gate_at_launch.json"
cp "$qualified/runtime_manifest.json" "$out/qualified_runtime_at_launch.json"
cp "$root/run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/gate.json" "$out/target_gate_at_launch.json"
cp "$root/docs/nonconvex_solver_benchmark_plan.md" "$out/plan_at_launch.md"
sha256sum "$out/"*.py "$out/"*.sbatch "$qualified/convergence_worker" > "$out/source_sha256.txt"
sha256sum "$root/src/libshell/TinyADHessian_Bilayer.hpp" "$root/src/libshell/ShellEquilibriumSolver.hpp" "$root/external/hlbfgs/HLBFGS.cpp" > "$out/production_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; lscpu; env | grep -E '^(OMP_|MKL_|OPENBLAS_|BLIS_|LD_)'; } > "$out/environment.txt"
cpus=$(python -c 'import os; a=sorted(os.sched_getaffinity(0)); assert len(a)>=4; print(",".join(map(str,a[:4])))')
echo "$cpus" > "$out/pinned_cpus.txt"
taskset -c "$cpus" python "$out/run_profiled_zigzag_pilot.py" --out "$out" --root "$root" --composed
sha256sum -c "$out/source_sha256.txt" "$out/production_sha256.txt" > "$out/final_hash_check.txt"
