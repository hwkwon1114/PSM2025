"""
Analyse the tolerance x mesh grid by the shape of the final curvature field.

For each mesh, the tightest-tolerance run is the reference and the looser runs are
compared against it element-wise (same mesh, so no interpolation is needed). This
isolates SOLVER error. Comparing the reference runs across meshes instead isolates
DISCRETIZATION error, and the useful tolerance for a given mesh is the loosest one whose
solver error sits below that mesh's discretization error -- there is no point converging
far past what the mesh itself can represent.

Curvature norms are area weighted, since the triangles are not uniform.

Usage:
    python python/curvature_sensitivity.py /gpfs/projects/p33082/psm_scaling
"""

import glob
import os
import re
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_final_dump(rundir):
    """Return (areas, gauss, mean) from the last swelling stage in a run directory."""
    cands = [f for f in glob.glob(os.path.join(rundir, "*growth_[0-9]*.vtp"))
             if "energies" not in os.path.basename(f)]
    if not cands:
        return None

    def idx(f):
        m = re.search(r"growth_(\d+)\.vtp$", f)
        return int(m.group(1)) if m else -1

    fn = max(cands, key=idx)
    if idx(fn) == 0:          # only the pre-solve dump exists: run produced nothing
        return None

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fn)
    reader.Update()
    pd = reader.GetOutput()

    cd = pd.GetCellData()
    if cd.GetArray("gauss") is None or cd.GetArray("mean") is None:
        return None
    gauss = vtk_to_numpy(cd.GetArray("gauss")).astype(float)
    meanc = vtk_to_numpy(cd.GetArray("mean")).astype(float)

    pts = vtk_to_numpy(pd.GetPoints().GetData()).astype(float)
    polys = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:]
    v0, v1, v2 = pts[polys[:, 0]], pts[polys[:, 1]], pts[polys[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    return areas, gauss, meanc, os.path.basename(fn), idx(fn)


def area_norm(field, areas, order=2):
    """Area-weighted L2 or Linf norm."""
    if order == np.inf:
        return np.abs(field).max()
    return np.sqrt(np.sum(areas * field**2) / np.sum(areas))


def parse_cost(workdir, res, tol):
    """Pull iteration count and wall time from the per-cell summary line."""
    fn = os.path.join(workdir, "grid_%s_%s.txt" % (res, tol))
    if not os.path.exists(fn):
        return None, None, None
    parts = open(fn).read().split()
    # res tol nverts nstages totiters lastg seconds rc=
    try:
        return int(parts[4]), float(parts[5]), int(parts[6])
    except (IndexError, ValueError):
        return None, None, None


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else "/gpfs/projects/p33082/psm_scaling"

    cells = {}
    for d in sorted(glob.glob(os.path.join(workdir, "grid_res*_tol*"))):
        m = re.search(r"grid_res([0-9.e-]+)_tol([0-9.e-]+)$", os.path.basename(d))
        if not m:
            continue
        res, tol = m.group(1), m.group(2)
        data = read_final_dump(d)
        if data is None:
            print("  [skip] %s : no usable final dump" % os.path.basename(d))
            continue
        cells[(res, tol)] = data

    if not cells:
        print("no completed grid cells found in %s" % workdir)
        return

    meshes = sorted({r for r, _ in cells}, key=float, reverse=True)

    for res in meshes:
        # A cell is only comparable if it reached the same final swelling stage as the
        # others; a run still in progress (or truncated) has a lower stage index and its
        # curvature field describes a partially swollen shape, not the final one.
        stages = {t: cells[(res, t)][4] for r, t in cells if r == res}
        target = max(stages.values())
        usable = sorted([t for t, s in stages.items() if s == target], key=float)
        dropped = [t for t, s in stages.items() if s != target]
        if dropped:
            print()
            print("  [mesh %s] ignoring %s : reached stage %s, not %d"
                  % (res, ", ".join(dropped),
                     ", ".join(str(stages[t]) for t in dropped), target))
        if len(usable) < 2:
            print("  [mesh %s] fewer than 2 comparable cells at stage %d -- skipping"
                  % (res, target))
            continue

        tols = usable
        ref_tol = tols[0]                      # tightest tolerance is the reference
        areas, g_ref, h_ref, ref_file, _ = cells[(res, ref_tol)]

        print()
        print("=" * 88)
        print("mesh res=%s   (%d faces)   reference tolerance = %s   [%s]"
              % (res, len(areas), ref_tol, ref_file))
        print("=" * 88)
        print("%-10s %-10s %-12s %-12s %-12s %-10s" %
              ("solvertol", "iters", "relL2(H)", "relLinf(H)", "relL2(K)", "wall_s"))
        print("-" * 88)

        hnorm = area_norm(h_ref, areas)
        gnorm = area_norm(g_ref, areas)

        for tol in sorted(tols, key=float, reverse=True):
            a, g, h, _, _ = cells[(res, tol)]
            if len(a) != len(areas):
                print("%-10s  (different mesh size -- skipped)" % tol)
                continue
            it, _lastg, secs = parse_cost(workdir, res, tol)
            rel_l2_h = area_norm(h - h_ref, areas) / hnorm if hnorm > 0 else float("nan")
            rel_li_h = area_norm(h - h_ref, areas, np.inf) / (np.abs(h_ref).max() or 1)
            rel_l2_g = area_norm(g - g_ref, areas) / gnorm if gnorm > 0 else float("nan")
            print("%-10s %-10s %-12.4e %-12.4e %-12.4e %-10s"
                  % (tol, it if it is not None else "?", rel_l2_h, rel_li_h,
                     rel_l2_g, secs if secs is not None else "?"))

    # discretization error: compare the reference runs across meshes using
    # mesh-independent integral measures, since the meshes differ element for element
    print()
    print("=" * 88)
    print("across meshes, tightest tolerance only (mesh-independent integrals)")
    print("=" * 88)
    print("%-10s %-8s %-16s %-16s %-16s" %
          ("res", "faces", "area", "int|H|dA", "int|K|dA"))
    print("-" * 88)
    for res in meshes:
        stages = {t: cells[(res, t)][4] for r, t in cells if r == res}
        target = max(stages.values())
        tols = sorted([t for t, s2 in stages.items() if s2 == target], key=float)
        if not tols:
            continue
        areas, g, h, _, st = cells[(res, tols[0])]
        print("%-10s %-8d %-16.6e %-16.6e %-16.6e   (stage %d, tol %s)"
              % (res, len(areas), areas.sum(),
                 np.sum(areas * np.abs(h)), np.sum(areas * np.abs(g)), st, tols[0]))


if __name__ == "__main__":
    main()
