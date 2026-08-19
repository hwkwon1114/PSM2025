#!/bin/bash
#SBATCH --job-name=zz_cont
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/pxl1051/build_verify_psm/zzcont-%j.out
#SBATCH --error=/home/pxl1051/build_verify_psm/zzcont-%j.out

set -eo pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WT=/gpfs/projects/p33082/PSM2025-zigzag
RUN=$WT/run/zz_seq_demo

echo "==== growth-continuation study: same cumulative growth, finer increments, tol 1e-8 ===="
echo "==== truth reference (already run): 6 x 0.05 @ tol 1e-12 -> max_U3=72.2mm, 30 min ===="

run_case () {  # name json tol
  local name=$1 json=$2 tol=$3
  local d="$RUN/$name"; mkdir -p "$d"; cd "$d"
  /usr/bin/time -f "WALL %e s" "$WT/bin/shell" \
    -sim bilayer_growth -case custom -geometry rectangle \
    -lx 0.13 -ly 0.16 -res 0.008 -h_total 0.0005 \
    -growth_type zigzag_sequence -cycle_file "$RUN/$json" \
    -enable_passE false -nsteps 1 -tol "$tol" -minimizer hlbfgs \
    -max_iter 50000 -basename "$name" -export_stl false > run.log 2>&1 || true
  local wall=$(grep -oE "WALL [0-9.]+ s" run.log | tail -1)
  local last=$(awk -F, 'END{printf "final max_U3=%.4e  E=%.4e", $18, $19}' \
      "$d/bilayer_zigzag_sequence_summary.csv" 2>/dev/null)
  echo "RESULT  $name  ->  $last  [$wall]"
  # compact trajectory: max_U3 every few cycles (the continuation curve)
  awk -F, 'NR>1 && (NR-1)%5==0 {printf "    traj cycle %s: max_U3=%.4e\n", $1, $18}' \
      "$d/bilayer_zigzag_sequence_summary.csv" 2>/dev/null | tail -14
}

run_case cont12 zigzag_cont12.json 1e-8
run_case cont30 zigzag_cont30.json 1e-8
run_case cont60 zigzag_cont60.json 1e-8
echo "==== ALLDONE ===="
