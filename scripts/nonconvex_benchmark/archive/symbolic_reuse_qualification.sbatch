#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=symbolic_reuse_gate
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/symbolic_reuse_gate_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
frozen="$root/run/nonconvex_solver_benchmark/development_7546843/source"
out="$root/run/nonconvex_solver_benchmark/symbolic_reuse_gate_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=false CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
mkdir -p "$out/source/test/diagnostics" "$out/source/test/testshell"
cp "$frozen/test/diagnostics/"*.hpp "$out/source/test/diagnostics/"
cp "$frozen/test/testshell/Test_NonconvexScreen.cpp" "$out/source/test/testshell/"
cp "$root/test/diagnostics/BenchmarkNewtonFactorization.hpp" "$root/test/diagnostics/NonconvexBenchmark.hpp" "$root/test/diagnostics/QualifySymbolicReuse.cpp" "$out/source/test/diagnostics/"
cp "$root/python/qualify_symbolic_reuse.py" "$root/python/run_nonconvex_screen.py" "$root/python/qualify_cholmod_integration.py" "$out/"
sha256sum "$out/source/test/diagnostics/"* "$out/source/test/testshell/"* "$out/"*.py "$frozen/lib/"*.a "$frozen/build/compile_commands.json" > "$out/source_sha256.txt"
sha256sum "$root/src/libshell/TinyADHessian_Bilayer.hpp" "$root/src/libshell/ShellEquilibriumSolver.hpp" "$root/external/hlbfgs/HLBFGS.cpp" > "$out/production_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; } > "$out/environment.txt"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
echo "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" python "$out/qualify_symbolic_reuse.py" --out "$out" --root "$root"
sha256sum -c "$out/production_sha256.txt"
