#!/usr/bin/env bash
# Coarse-side Putong-parity HLBFGS resolution sweep.

set -u -o pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
solver=${PSM_HLBFGS_SOLVER:-$repo_root/bin/shell}
config=${PSM_HLBFGS_CONFIG:-$repo_root/scripts/certification_benchmarks/putong_1step.json}
output_root=${PSM_HLBFGS_OUTPUT:-$repo_root/run/hlbfgs_resolution_sweep}
resolutions=${PSM_HLBFGS_RESOLUTIONS:-"0.08 0.06 0.05 0.04 0.035 0.03 0.025 0.02 0.015"}

mkdir -p "$output_root"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

for resolution in $resolutions; do
  resolution_tag=$(awk -v value="$resolution" 'BEGIN{printf "%03d", 1000*value}')
  case_name="r${resolution_tag}"
  case_dir="$output_root/$case_name"
  mkdir -p "$case_dir"

  if [[ -s "$case_dir/bilayer_zigzag_sequence_summary.csv" &&
        "${PSM_HLBFGS_FORCE:-0}" != "1" ]]; then
    echo "$case_name already complete"
    continue
  fi

  (
    cd "$case_dir"
    /usr/bin/time -f 'wall_seconds=%e' \
      "$solver" \
      -sim bilayer_growth -case custom -geometry rectangle \
      -lx 0.127 -ly 0.1524 -res "$resolution" -h_total 0.0006 \
      -growth_type zigzag_sequence -cycle_file "$config" \
      -enable_passE false -nsteps 1 \
      -tol 1e-12 -minimizer hlbfgs \
      -max_iter 50000 -seed_escape false -export_stl false > run.log 2>&1
  )
  status=$?

  if [[ $status -eq 0 && -s "$case_dir/bilayer_zigzag_sequence_summary.csv" ]]; then
    energy=$(awk -F, 'END{print $19}' "$case_dir/bilayer_zigzag_sequence_summary.csv")
    u3=$(awk -F, 'END{print $18-$17}' "$case_dir/bilayer_zigzag_sequence_summary.csv")
    wall=$(grep -oE 'wall_seconds=[0-9.]+' "$case_dir/run.log" | tail -1)
    echo "$case_name complete: U3_span=$u3 energy=$energy $wall"
  else
    echo "$case_name failed: rc=$status (see $case_dir/run.log)"
  fi
done
