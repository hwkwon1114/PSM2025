#!/usr/bin/env bash
# No convergence fits. Bounded integration/restart checks on four quest10 CPUs:
# SLURM_CONSTRAINT=quest10 srun-here --cpus 4 --mem 16G --time 00:06:00 \
#   --env smcpp_vtk38 bash scripts/nonconvex_benchmark/parallel_hessian_solver_qualification.sh
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_CPUS_PER_TASK:-0} -eq 4 ]]
: "${CONDA_PREFIX:?Activate smcpp_vtk38}"
frozen="$root/run/nonconvex_solver_benchmark/development_7546843/source"
qualified="$root/run/nonconvex_solver_benchmark/guarded_hessian_gate_7637624"
out="$root/run/nonconvex_solver_benchmark/parallel_solver_gate_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'printf "%s\n" "$?" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=4 OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES=""
model=$(lscpu | awk -F: '/^Model name:/{gsub(/^[ \t]+/,"",$2);print $2;exit}')
reference=$(awk -F: '/^Model name:/{gsub(/^[ \t]+/,"",$2);print $2;exit}' "$root/run/nonconvex_solver_benchmark/development_7516445/environment.txt")
[[ -n "$model" && "$model" == "$reference" ]]
mkdir "$out/source"
cp -r "$qualified/source/test" "$out/source/test"
for name in NonconvexBenchmark.hpp ShellBenchmarkProblem.hpp LocalDihedralCachedTinyADHessian_Bilayer.hpp RobustCurvatureHandoff.hpp BenchmarkNewtonFactorization.hpp CholmodFrozenFactorization.hpp ShiftedNewtonFactorization.hpp; do
 cp "$root/test/diagnostics/$name" "$out/source/test/diagnostics/"
done
cp "$root/test/testshell/Test_NonconvexScreen.cpp" "$out/source/test/testshell/"
for name in ConvergenceBenchmark.cpp QualifyHybridShell.cpp HybridBenchmarkTransition.hpp QualifyParallelHessian.cpp RobustCurvatureHandoff.hpp BenchmarkNewtonFactorization.hpp CholmodFrozenFactorization.hpp ShiftedNewtonFactorization.hpp; do
 cp "$root/test/diagnostics/$name" "$out/source/"
done
for name in qualify_parallel_hessian_solver run_convergence_benchmark run_nonconvex_screen qualify_cholmod_integration; do
 cp "$root/python/$name.py" "$out/"
done
cp "$0" "$out/"
cp "$root/docs/nonconvex_solver_benchmark_plan.md" "$out/plan_at_launch.md"
sha256sum "$out/source/"*.* "$out/source/test/diagnostics/"* "$out/source/test/testshell/"* \
 "$out/"*.py "$out/"*.sh "$frozen/lib/"*.a "$frozen/build/compile_commands.json" > "$out/source.sha256"
sha256sum "$root/src/libshell/TinyADHessian_Bilayer.hpp" "$root/src/libshell/ShellEquilibriumSolver.hpp" \
 "$root/external/hlbfgs/HLBFGS.cpp" > "$out/production.sha256"
{ hostname; date -Is; lscpu; env | grep -E '^(OMP_|SLURM_|CONDA_PREFIX|MKL_|OPENBLAS_|LD_)'; } > "$out/environment.txt"
timeout --signal=TERM --kill-after=5 330 python "$out/qualify_parallel_hessian_solver.py" --out "$out" --root "$root"
sha256sum --check "$out/source.sha256" "$out/production.sha256" > "$out/source_check.txt"
printf 'PARALLEL_SOLVER_GATE_COMPLETE; no trajectory pilot\n' > "$out/complete.txt"
