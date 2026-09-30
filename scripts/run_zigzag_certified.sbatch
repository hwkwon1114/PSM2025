#!/bin/bash
#SBATCH --job-name=zz_prod
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/zigzag_production/zzprod_%j.out
#
# Production zigzag: continuation + exact-Hessian certification/escape.
#
# Usage:
#   sbatch run_zigzag_certified.sbatch                      # default: cont30 @ res 0.03, tol 1e-12
#   sbatch run_zigzag_certified.sbatch CYCLE.json RES NAME [SEEDMAX LX LY HTOT TOL]
#   sbatch run_zigzag_certified.sbatch check                # 3-case convergence suite
#
# TWO REGIMES -- pick the tolerance to match (positional arg 8, TOL; default 1e-12):
#   * STRESS / large-growth (accumulated growth ~O(0.1-1), deep post-buckling):
#     TOL=1e-8 + seed_escape. The trap here is NEGATIVE CURVATURE (saddles), which
#     the exact-Hessian escape fixes cheaply -- 204 s vs a 1833 s tol-1e-12 grind.
#   * PHYSICAL / small-growth (Putong parity, gtop ~0.003, one english-wheel pass):
#     TOL=1e-12. The trap here is a FLAT PLATEAU -- the gradient toward the buckled
#     shape sits below 1e-8, so the solve stops flat and certifies a bogus "minimum"
#     (nothing negative to escape). summary_row prints a WARNING when it detects this
#     (certified MINIMUM at max|z| < 1e-4 m); re-run that case with TOL=1e-12.
#
# Recipe justified by the certification benchmarks (bench_escape/bench_tolconv):
#   - growth split into many small repeats (continuation) at MODERATE -tol 1e-8;
#     tol 1e-6 and looser terminates on the FLAT state (max_U3 ~ 2e-4, step 0).
#   - -seed_escape: exact-TinyAD-Hessian certification of the final state; while
#     negative curvature remains, seed along the buckling mode and re-minimize.
#     Job 9853638: certified MINIMUM in 204 s vs 1833 s for a tol-1e-12 grind
#     that still ended uncertified.
#   - KNOWN CAVEAT (job 9853638): the endpoint is branch-sensitive -- cont60
#     reached a lower-energy state than cont30, and the mesh is not converged
#     (res 0.0057 shifts the signed curvature ~13%). For production numbers,
#     run "check" mode on the target case and confirm the rows agree.
set -o pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38   # activation scripts use unbound vars; enable -u only after
set -u

WT=/gpfs/projects/p33082/PSM2025-zigzag
CFG=$WT/trajectories
RUN=$WT/run/zigzag_production
mkdir -p "$RUN"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

run_case () {  # name res cycle_json
  local name=$1 res=$2 cyc status
  cyc=$(realpath "$3")
  local d="$RUN/$name"; rm -rf "$d"; mkdir -p "$d"; cd "$d"
  echo "==== CASE $name (res=$res cycle=$(basename "$cyc")) ===="
  /usr/bin/time -f "WALL %e s" "$WT/bin/shell" \
    -sim bilayer_growth -case custom -geometry rectangle \
    -lx "${LX:-0.13}" -ly "${LY:-0.16}" -res "$res" -h_total "${HTOT:-0.0005}" \
    -growth_type zigzag_sequence -cycle_file "$cyc" \
    -enable_passE false -nsteps 1 \
    -tol "${TOL:-1e-12}" -minimizer hlbfgs -max_iter 50000 \
    -seed_escape true -seed_max "${SEEDMAX:-12}" \
    -basename "$name" -export_stl true > run.log 2>&1 || {
      status=$?
      echo "==== ${name}_FAILED status=$status ===="
      tail -40 run.log
      return "$status"
    }
  grep -E "certify_final|seed_escape" run.log | tail -4
  echo "==== ${name}_DONE ===="
}

summary_row () {  # name res
  local name=$1 res=$2
  local d="$RUN/$name"   # separate line: $name expands before `local` assigns
  local wall lam verdict E u3 nf sh ah
  wall=$(grep -oE "WALL [0-9.]+ s" "$d/run.log" | tail -1 | awk '{print $2"s"}')
  # POST-escape values come from the last certify_final line; the sequence CSV
  # is written BEFORE the escape loop and is only the fallback.
  lam=$(grep -oE "lam_min=[+-][0-9.e-]+" "$d/run.log" | tail -1 | cut -d= -f2)
  verdict=$(grep -E "certify_final.*-> " "$d/run.log" | tail -1 | sed 's/.*-> //')
  E=$(grep -oE "certify_final\] esc=[0-9]+  E=[0-9.e-]+" "$d/run.log" | tail -1 | grep -oE "E=[0-9.e-]+" | cut -d= -f2)
  u3=$(grep -oE "max\|z\|=[0-9.e-]+" "$d/run.log" | tail -1 | cut -d= -f2)
  local csv="$d/bilayer_zigzag_sequence_summary.csv"
  [ -z "$E" ]  && E=$(awk -F, 'END{printf "%.6e",$19}' "$csv" 2>/dev/null || echo -)
  [ -z "$u3" ] && u3=$(awk -F, 'END{printf "%.4e",$18}' "$csv" 2>/dev/null || echo -)
  # curvature metrics from the last cycle's *_final.vtp (deformed geometry,
  # carries the 'mean' cell array; there is no final_curv.vtp in sequence mode)
  read -r nf sh ah <<< "$(python3 - "$d" <<'PY'
import sys, glob, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
fn = sorted(glob.glob(f"{sys.argv[1]}/bilayer_zigzag_sequence_cycle_*_final.vtp"))
if not fn: print("- - -"); raise SystemExit
r = vtk.vtkXMLPolyDataReader(); r.SetFileName(fn[-1]); r.Update(); pd = r.GetOutput()
p = v2n(pd.GetPoints().GetData()); t = v2n(pd.GetPolys().GetData()).reshape(-1,4)[:,1:]
a = 0.5*np.linalg.norm(np.cross(p[t[:,1]]-p[t[:,0]], p[t[:,2]]-p[t[:,0]]), axis=1)
H = v2n(pd.GetCellData().GetArray('mean'))
print(len(a), "%+.4e" % np.sum(a*H), "%.4e" % np.sum(a*np.abs(H)))
PY
)"
  printf "%-9s %-8s %-7s %-13s %-12s %-12s %-12s %-11s %-8s %s\n" \
    "$name" "$res" "$nf" "${E:--}" "${u3:--}" "$sh" "$ah" "${lam:--}" "${wall:--}" "${verdict:--}"

  # FLAT-PLATEAU GUARD: a "certified minimum" at essentially zero deflection is the
  # physical-regime trap -- at small total growth the gradient toward the buckled
  # shape sits below tol 1e-8, so hlbfgs stops on the flat state and certify sees
  # non-negative curvature (nothing to escape). max|z| < 1e-4 m on a ~0.15 m panel
  # is flat. The fix is a tighter tolerance (Putong's setting), not more escapes.
  case "$verdict" in
    MINIMUM*)
      if awk "BEGIN{exit !(${u3:-1} < 1e-4)}" 2>/dev/null; then
        echo "  !! WARNING [$name]: certified MINIMUM but max|z|=${u3} m is essentially FLAT."
        echo "  !! Likely a tol-plateau stall, not the physical shape. Re-run with TOL=1e-12"
        echo "  !! (8th positional arg), e.g.: sbatch $0 ${CYCLE:-CYCLE.json} $res $name ${SEEDMAX:-12} ${LX:-0.127} ${LY:-0.1524} ${HTOT:-0.0006} 1e-12"
      fi
      ;;
  esac
}

summary_header () {
  echo ""
  echo "==== SUMMARY (post-escape values; rows should agree if converged) ===="
  printf "%-9s %-8s %-7s %-13s %-12s %-12s %-12s %-11s %-8s %s\n" \
    case res nfaces E "max|z|" sumHdA sumAbsHdA lam_min wall verdict
}

MODE=${1:-prod}
if [ "$MODE" = check ]; then
  run_case prod    0.008  "$CFG/zigzag_cont30.json" || exit $?
  run_case contref 0.008  "$CFG/zigzag_cont60.json" || exit $?
  run_case meshref 0.0057 "$CFG/zigzag_cont30.json" || exit $?
  summary_header
  summary_row prod    0.008
  summary_row contref 0.008
  summary_row meshref 0.0057
else
  CYCLE=${1:-$CFG/zigzag_cont30.json}
  RES=${2:-0.03}
  NAME=${3:-prod}
  SEEDMAX=${4:-12}   # max certified saddle escapes (raise if a run stops at the cap)
  LX=${5:-0.13}      # panel geometry; Putong-parity runs pass 0.127 0.1524 0.0006
  LY=${6:-0.16}
  HTOT=${7:-0.0005}
  TOL=${8:-1e-12}    # validated physical-regime default; large-growth studies may override with 1e-8
  run_case "$NAME" "$RES" "$CYCLE" || exit $?
  summary_header
  summary_row "$NAME" "$RES"
fi
echo "==== ALLDONE ===="
