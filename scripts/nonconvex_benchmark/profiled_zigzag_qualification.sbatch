#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=profiled_zigzag_gate
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/profiled_zigzag_gate_%j.out
set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
root=/gpfs/projects/p33082/PSM2025-zigzag
frozen="$root/run/nonconvex_solver_benchmark/development_7546843/source"
base="$root/run/nonconvex_solver_benchmark/local_dihedral_solver_gate_7547943/preflight/bundle"
out="$root/run/nonconvex_solver_benchmark/profiled_zigzag_gate_${SLURM_JOB_ID}"
mkdir "$out"
exec > >(tee "$out/job.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit_code.txt"' EXIT
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=false CHOLMOD_USE_GPU=0 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
mkdir "$out/source"
cp "$root/test/diagnostics/PrepareProfiledZigzag.cpp" "$out/source/"
cp "$root/python/qualify_profiled_zigzag.py" "$root/python/run_nonconvex_screen.py" "$root/python/qualify_cholmod_integration.py" "$root/python/visualize_solver_characteristics.py" "$root/python/analyze_sequence_ablation_v2.py" "$out/"
cp "$root/run/zigzag_abc_preview/sequence_ortho_minus_0p5.json" "$out/sequence.json"
cp "$root/run/zigzag_abc_preview/optimization_request.md" "$out/request_at_launch.md"
sha256sum "$out/source/"* "$out/"*.py "$out/sequence.json" "$frozen/lib/"*.a "$frozen/build/compile_commands.json" "$frozen/src/libshell/"*.hpp > "$out/source_sha256.txt"
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; } > "$out/environment.txt"
cpu=$(python -c 'import os;print(min(os.sched_getaffinity(0)))')
echo "$cpu" > "$out/pinned_cpu.txt"
taskset -c "$cpu" python "$out/qualify_profiled_zigzag.py" --out "$out" --frozen "$frozen" --baseline "$base"
