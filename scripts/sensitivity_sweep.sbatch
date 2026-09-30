#!/bin/bash
#SBATCH --job-name=zz_sens
#SBATCH --account=p33082
#SBATCH --partition=short
#SBATCH --array=0-29%10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:55:00
#SBATCH --output=/gpfs/projects/p33082/PSM2025-zigzag/run/sensitivity/sens_%A_%a.out
#
# 3-factor sensitivity study on the PHYSICAL (Putong-parity) english-wheel case.
# Factors: tolerance x resolution x HLBFGS certification mode (5 x 3 x 2 = 30 runs).
#   tol  : 1e-6 1e-8 1e-10 1e-12 1e-14   (spans the flat-stall cliff -> convergence)
#   res  : 0.012 0.008 0.0057            (mesh convergence, ~2x faces per step)
#   alg  : hlbfgs | certified
#          hlbfgs    = plain HLBFGS
#          certified = HLBFGS + seeded exact-Hessian saddle escape
# Every run is post-certified (-certify_final true) so lam_min/verdict is uniform.
# Geometry: Putong parity  lx0.127 ly0.1524 h0.0006, zigzag lv140/9.13deg/N10/w10,
# single-step growth 0.003 (putong_1step.json).
set -o pipefail
source ~/miniforge/etc/profile.d/conda.sh
conda activate smcpp_vtk38
set -u
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WT=/gpfs/projects/p33082/PSM2025-zigzag
CFG=$WT/trajectories
RUN=$WT/run/sensitivity
RES_DIR=$RUN/results
mkdir -p "$RES_DIR"

TOLS=(1e-6 1e-8 1e-10 1e-12 1e-14)
RESES=(0.012 0.008 0.0057)
ALGS=(hlbfgs certified)
NT=${#TOLS[@]}; NR=${#RESES[@]}

T=$SLURM_ARRAY_TASK_ID
TI=$(( T % NT ))
RI=$(( (T / NT) % NR ))
AI=$(( T / (NT * NR) ))
TOL=${TOLS[$TI]}; RES=${RESES[$RI]}; ALG=${ALGS[$AI]}

case "$ALG" in
  hlbfgs)    MINZ=hlbfgs;       EXTRA="" ;;
  certified) MINZ=hlbfgs; EXTRA="-seed_escape true -seed_max 12" ;;
esac

NAME="a-${ALG}_r${RES}_t${TOL}"
d="$RUN/$NAME"; rm -rf "$d"; mkdir -p "$d"; cd "$d" || exit 1
echo "task $T : alg=$ALG res=$RES tol=$TOL"

START=$(date +%s)
timeout 3000 "$WT/bin/shell" \
  -sim bilayer_growth -case custom -geometry rectangle \
  -lx 0.127 -ly 0.1524 -res "$RES" -h_total 0.0006 \
  -growth_type zigzag_sequence -cycle_file "$CFG/putong_1step.json" \
  -enable_passE false -nsteps 1 \
  -tol "$TOL" -minimizer "$MINZ" -max_iter 50000 \
  -certify_final true $EXTRA \
  -basename "$NAME" -export_stl false > run.log 2>&1
RC=$?
SECS=$(( $(date +%s) - START ))

# post-escape / certified values from the last certify_final line
LAM=$(grep -oE "lam_min=[+-][0-9.e-]+" run.log | tail -1 | cut -d= -f2)
VERD=$(grep -E "certify_final.*-> " run.log | tail -1 | sed 's/.*-> //' | tr ' ' '_')
[ "$RC" = 124 ] && VERD="TIMEOUT"
E=$(grep -oE "certify_final\] esc=[0-9]+  E=[0-9.e-]+" run.log | tail -1 | grep -oE "E=[0-9.e-]+" | cut -d= -f2)
[ -z "${E:-}" ] && E=$(awk -F, 'END{printf "%.6e",$19}' "$d/bilayer_zigzag_sequence_summary.csv" 2>/dev/null || echo -)

read -r NF U3 SH AH <<< "$(python3 - "$d" <<'PY'
import sys, glob, numpy as np, vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
fn = sorted(glob.glob(f"{sys.argv[1]}/bilayer_zigzag_sequence_cycle_*_final.vtp"))
if not fn: print("- - - -"); raise SystemExit
r = vtk.vtkXMLPolyDataReader(); r.SetFileName(fn[-1]); r.Update(); pd = r.GetOutput()
p = v2n(pd.GetPoints().GetData()); t = v2n(pd.GetPolys().GetData()).reshape(-1,4)[:,1:]
a = 0.5*np.linalg.norm(np.cross(p[t[:,1]]-p[t[:,0]], p[t[:,2]]-p[t[:,0]]), axis=1)
H = v2n(pd.GetCellData().GetArray('mean'))
print(len(a), "%.4e" % np.abs(p[:,2]).max(), "%+.4e" % np.sum(a*H), "%.4e" % np.sum(a*np.abs(H)))
PY
)"

# one parseable line per run (tab-separated) for the collation step
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$ALG" "$RES" "$TOL" "${NF:--}" "${E:--}" "${U3:--}" "${SH:--}" "${AH:--}" \
  "${LAM:--}" "${VERD:--}" "$SECS" > "$RES_DIR/res_$(printf '%02d' "$T").txt"
echo "DONE task $T : $(cat "$RES_DIR/res_$(printf '%02d' "$T").txt")"
