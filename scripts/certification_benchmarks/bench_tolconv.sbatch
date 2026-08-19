#!/bin/bash
#SBATCH --job-name=zz_tolconv
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/pxl1051/build_verify_psm/zztol-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/zztol-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WT=/gpfs/projects/p33082/PSM2025-zigzag
RUN=$WT/run/zz_seq_demo

echo "==== tolerance-convergence study (large-buckling recurring case, res 0.008) ===="

run_case () {  # name minimizer tol extra
  local name=$1 minz=$2 tol=$3; shift 3
  local d="$RUN/$name"; mkdir -p "$d"; cd "$d"
  /usr/bin/time -f "WALL %e s" "$WT/bin/shell" \
    -sim bilayer_growth -case custom -geometry rectangle \
    -lx 0.13 -ly 0.16 -res 0.008 -h_total 0.0005 \
    -growth_type zigzag_sequence -cycle_file "$RUN/zigzag_large.json" \
    -enable_passE false -nsteps 1 -tol "$tol" -minimizer "$minz" \
    -basename "$name" -export_stl false "$@" > run.log 2>&1 || true
  local wall=$(grep -oE "WALL [0-9.]+ s" run.log | tail -1)
  # cycle-6 (last) max_U3 (col 18) + total_energy (col 19) from the summary CSV
  local last=$(awk -F, 'END{printf "max_U3=%.6e  E=%.4e", $18, $19}' \
      "$d/bilayer_zigzag_sequence_summary.csv" 2>/dev/null)
  echo "RESULT  $name  (minz=$minz tol=$tol)  ->  $last  [$wall]"
}

# HLBFGS tolerance sweep (relative gradient tol)
run_case tol_hl_1e4  hlbfgs 1e-4  -max_iter 50000
run_case tol_hl_1e6  hlbfgs 1e-6  -max_iter 50000
run_case tol_hl_1e8  hlbfgs 1e-8  -max_iter 50000
run_case tol_hl_1e10 hlbfgs 1e-10 -max_iter 50000
run_case tol_hl_1e12 hlbfgs 1e-12 -max_iter 50000
# well-conditioned reference: exact-Hessian preconditioned HLBFGS at tight tol
run_case tol_pc_1e10 hlbfgs_precond 1e-10
echo "==== ALLDONE ===="
