#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=solver_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:13:00
#SBATCH --nodelist=qnode0025
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/repair_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
src="$root/run/nonconvex_solver_benchmark/development_7290186/source"
out="$root/run/nonconvex_solver_benchmark/preflight_repair_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/build_and_tests.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=false
cp "$src/test/testshell/Test_NonconvexScreen.cpp" "$out/original_failed_Test_NonconvexScreen.cpp"
cp "$root/test/testshell/Test_NonconvexScreen.cpp" "$src/test/testshell/Test_NonconvexScreen.cpp"
cp "$src/python/run_nonconvex_screen.py" "$out/original_runner.py"
cp "$root/python/run_nonconvex_screen.py" "$src/python/run_nonconvex_screen.py"
cp "$root/scripts/nonconvex_preflight_repair.sbatch" "$out/"
find "$src/src" "$src/external" "$src/test" "$src/python" "$src/cmake" "$src/CMakeLists.txt" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/source_after_fix_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; lscpu; } > "$out/environment.txt"
cmake --build "$src/build" --target testshell --parallel 1
sha256sum "$src/bin/testshell" > "$out/binary.sha256"
mkdir "$out/tests"
cd "$out/tests"
"$src/bin/testshell" --gtest_filter='Benchmark*:ShellEquilibriumProjector.*:ShellEquilibriumSteihaug.*' --gtest_output=xml:tests.xml | tee tests.log
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
taskset -c "$cpu" python "$src/python/run_nonconvex_screen.py" preflight --binary "$src/bin/testshell" --root "$out/preflight"
printf 'SCREEN_PREFLIGHT_PASSED; scientific fits still not launched\n'
