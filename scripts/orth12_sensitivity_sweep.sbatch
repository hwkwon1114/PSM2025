#!/bin/bash
#SBATCH --job-name=o12_sens
#SBATCH --account=p33082
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-24%5
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/orth12_sensitivity/slurm_%A_%a.out
#SBATCH --error=/gpfs/projects/p33082/PSM2025-zigzag/run/orth12_sensitivity/slurm_%A_%a.out

set -o pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

repo_root=/gpfs/projects/p33082/PSM2025-zigzag
output_root=$repo_root/run/orth12_sensitivity
cycle_file=$repo_root/scripts/certification_benchmarks/orthogonal_opposite_sides_12cycle.json

tolerances=(1e-10 3e-11 1e-11 3e-12 1e-12)
resolutions=(0.03 0.025 0.02 0.015 0.01)

tol_index=$((SLURM_ARRAY_TASK_ID % ${#tolerances[@]}))
res_index=$((SLURM_ARRAY_TASK_ID / ${#tolerances[@]}))
tol=${tolerances[$tol_index]}
res=${resolutions[$res_index]}
res_tag=$(awk -v value="$res" 'BEGIN{printf "%03d",1000*value}')
tol_tag=${tol//-/m}
case_name="r${res_tag}_t${tol_tag}"
case_dir=$output_root/$case_name
mkdir -p "$case_dir"
cd "$case_dir" || exit 1

if [[ -s result.tsv ]]; then
  previous_status=$(awk -F'\t' 'END{print $4}' result.tsv)
  previous_verdict=$(awk -F'\t' 'END{print $9}' result.tsv)
  previous_final=$(awk -F'\t' 'END{print $10}' result.tsv)
  if [[ "$previous_status" = "0" && "$previous_verdict" = MINIMUM_* && -n "$previous_final" && -s "$previous_final" ]]; then
    echo "SKIP $case_name: completed result already exists"
    exit 0
  fi
fi

rm -f result.tsv run.log argumentparser.log bilayer_zigzag_sequence_*
start=$(date +%s)
timeout 14000 "$repo_root/bin/shell" \
  -sim bilayer_growth -case custom -geometry rectangle \
  -lx 0.13 -ly 0.16 -res "$res" -h_total 0.0005 \
  -growth_type zigzag_sequence -cycle_file "$cycle_file" \
  -enable_passE false -nsteps 1 \
  -tol "$tol" -minimizer hlbfgs -max_iter 50000 \
  -certify_final true \
  -basename "$case_name" -export_stl false > run.log 2>&1
status=$?
wall_seconds=$(( $(date +%s) - start ))

energy=$(grep -oE 'certify_final\] esc=[0-9]+  E=[0-9.e+-]+' run.log | tail -1 | grep -oE 'E=[0-9.e+-]+' | cut -d= -f2)
gradient=$(grep -oE '\|g\|=[0-9.e+-]+' run.log | tail -1 | cut -d= -f2)
lambda_min=$(grep -oE 'lam_min=[+-][0-9.e-]+' run.log | tail -1 | cut -d= -f2)
verdict=$(grep -E 'certify_final.*-> ' run.log | tail -1 | sed 's/.*-> //' | tr ' ' '_')
final_file=$(find . -maxdepth 1 -name 'bilayer_zigzag_sequence_cycle_012_*_final.vtp' -printf '%f\n' | sort | tail -1)

if [[ "$status" = 124 ]]; then
  verdict=TIMEOUT
elif [[ "$status" = 0 && "$verdict" != MINIMUM_* ]]; then
  status=3
  [[ -n "$verdict" ]] || verdict=FAILED
elif [[ -z "$verdict" ]]; then
  verdict=FAILED
fi
if [[ -z "$energy" && -s bilayer_zigzag_sequence_summary.csv ]]; then
  energy=$(awk -F, 'END{printf "%.6e",$19}' bilayer_zigzag_sequence_summary.csv)
fi

printf 'task_id\tresolution\ttolerance\tstatus\twall_seconds\tenergy\tgradient\tlambda_min\tverdict\tfinal_file\n' > result.tsv
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$SLURM_ARRAY_TASK_ID" "$res" "$tol" "$status" "$wall_seconds" \
  "${energy:--}" "${gradient:--}" "${lambda_min:--}" "$verdict" "${final_file:--}" >> result.tsv

echo "DONE $case_name status=$status wall=${wall_seconds}s energy=${energy:--} gradient=${gradient:--} lambda_min=${lambda_min:--} verdict=$verdict"
exit "$status"
