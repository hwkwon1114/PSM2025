#!/bin/bash
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --job-name=nonconvex_screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/nonconvex_solver_benchmark/screen_%j.out
set -eo pipefail
start=$SECONDS
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
: "${PREFLIGHT:?Supply verified preflight directory}"
root=/gpfs/projects/p33082/PSM2025-zigzag
out="$root/run/nonconvex_solver_benchmark/screen_${SLURM_JOB_ID}"
meta="$root/run/nonconvex_solver_benchmark/submission_${SLURM_JOB_ID}"
mkdir "$meta"
cp "$root/scripts/nonconvex_screen.sbatch" "$meta/"
cp "$root/python/plot_nonconvex_screen.py" "$meta/"
cp "$PREFLIGHT/gate.json" "$meta/"
cp "$root/reports/nonconvex_solver_benchmark_plan.md" "$meta/protocol_at_launch.md"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=false MPLBACKEND=Agg
{ hostname; date -Is; scontrol show job "$SLURM_JOB_ID"; lscpu; conda list; } > "$meta/environment.txt"
binary=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["binary"])' "$PREFLIGHT/gate.json")
runner="$(dirname "$(dirname "$binary")")/python/run_nonconvex_screen.py"
cp "$runner" "$meta/run_nonconvex_screen.py"
sha256sum "$binary" "$runner" "$meta/plot_nonconvex_screen.py" > "$meta/sha256.txt"
cpu=$(python -c 'import os; print(min(os.sched_getaffinity(0)))')
echo "SCREEN_ROOT=$out CPU=$cpu PREFLIGHT=$PREFLIGHT"
taskset -c "$cpu" python "$meta/run_nonconvex_screen.py" screen --preflight "$PREFLIGHT" --root "$out"
# Fixed batch allocation bounds all work, not only solver time. Rendering is
# optional if little budget remains; never extend/restart a fit to fill a figure.
remaining=$((7200-(SECONDS-start)))
if (( remaining > 65 )); then
 if timeout 10 taskset -c "$cpu" python "$meta/plot_nonconvex_screen.py" --self-test && \
    timeout 50 taskset -c "$cpu" python "$meta/plot_nonconvex_screen.py" "$out"; then
  echo 'Figures exported; visual review still pending' > "$meta/rendering_status.txt"
 else
  echo 'Rendering failed/partial; numerical records preserved, no retry' > "$meta/rendering_status.txt"
 fi
else
 echo 'Rendering deferred: global allocation nearly exhausted' > "$meta/rendering_deferred.txt"
fi
