"""
Locate the buckling threshold of the flat disk by scanning the swelling fraction.

For each swelling fraction t the growth stretches are ramped linearly from 1 to their
target (matching MetricContinuation::interpolateFromIsotropic, which is what
assignGrowthToMetric does in run_basic_disk). At each t we

  1. relax the in-plane displacement to get the pre-buckling base state,
  2. compute the lowest eigenvalues of the out-of-plane Hessian block.

lambda_0 crossing zero is the genuine bifurcation point: below it the flat sheet is a
local minimum and there is nothing to seed; above it, the eigenvector at threshold is
the mode that actually initiates buckling.

Run from the repo root:  python python/swelling_scan.py
"""

import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bindings import (  # noqa: E402
    build_validation_disk,
    hessian_operator,
    out_of_plane_rigid_modes,
    relax_in_plane,
)

RES = 24
NEV = 6


def spectrum_at(t, res=RES, nev=NEV, want_vectors=False):
    """Lowest out-of-plane Hessian eigenvalues at swelling fraction t."""
    s = build_validation_disk(res=res, swelling=t)
    nV = s.n_vertices
    x_base, _ = relax_in_plane(s)

    idx = np.concatenate([np.arange(2 * nV, 3 * nV), np.arange(3 * nV, s.n_dofs)])
    Q = out_of_plane_rigid_modes(s, x_base)
    H = hessian_operator(s, x_base, idx=idx, deflate=Q)

    out = eigsh(H, k=nev, which="SA", tol=1e-9, maxiter=40000,
                return_eigenvectors=want_vectors)
    if want_vectors:
        vals, vecs = out
        order = np.argsort(vals)
        return s, x_base, idx, vals[order], vecs[:, order]
    return np.sort(out)


def main():
    print(f"mesh resolution res={RES}, tracking {NEV} lowest out-of-plane modes")
    print()

    # The swelling fractions run_basic_disk steps through start at 0.01, which already
    # turns out to be past threshold, so extend the scan well below the first stage.
    coarse = [1e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3,
              0.01, 0.05, 0.1, 0.3, 0.6, 1.0]

    print(f"{'t':>6}  {'lambda_0':>13}  {'lambda_1':>13}  {'#negative':>9}")
    print("-" * 50)

    results = []
    for t in coarse:
        vals = spectrum_at(t)
        n_neg = int((vals < 0).sum())
        results.append((t, vals[0], n_neg))
        print(f"{t:6.3f}  {vals[0]:+13.6e}  {vals[1]:+13.6e}  {n_neg:9d}"
              + ("   <-- unstable" if vals[0] < 0 else ""))

    # bracket the first sign change of lambda_0
    bracket = None
    for (t_lo, l_lo, _), (t_hi, l_hi, _) in zip(results[:-1], results[1:]):
        if l_lo > 0.0 >= l_hi:
            bracket = (t_lo, t_hi)
            break

    print()
    if bracket is None:
        if results[0][1] < 0:
            print("lambda_0 is already negative at the smallest swelling scanned:")
            print(f"  the disk is unstable below t = {coarse[0]}; rerun with smaller t")
        else:
            print("no sign change found: the flat state stays stable over this range")
        return

    lo, hi = bracket
    print(f"bifurcation bracketed in t = [{lo}, {hi}] -- bisecting")
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        lam0 = spectrum_at(mid)[0]
        print(f"   t = {mid:.6f}   lambda_0 = {lam0:+.6e}")
        if lam0 > 0.0:
            lo = mid
        else:
            hi = mid

    t_crit = 0.5 * (lo + hi)
    print()
    print(f"critical swelling fraction  t_crit ~ {t_crit:.4f}")

    # the mode that initiates buckling, evaluated just past threshold
    t_post = min(1.0, t_crit * 1.05)
    s, x_base, idx, vals, vecs = spectrum_at(t_post, want_vectors=True)
    nV = s.n_vertices
    print(f"spectrum just past threshold (t = {t_post:.4f}):")
    for i, lam in enumerate(vals):
        print(f"   lambda[{i}] = {lam:+.6e}" + ("   <-- unstable" if lam < 0 else ""))
    n_neg = int((vals < 0).sum())
    print(f"   negative modes at threshold: {n_neg} of {NEV} tracked")

    mode = np.zeros(s.n_dofs)
    mode[idx] = vecs[:, 0]
    z = mode[2 * nV:3 * nV]
    verts = s.vertices()
    theta = np.arctan2(verts[:, 1], verts[:, 0])

    # dominant angular wavenumber of the initiating mode
    print()
    print("   angular content of the initiating mode:")
    for n in range(0, 7):
        amp = np.hypot(np.sum(z * np.cos(n * theta)), np.sum(z * np.sin(n * theta)))
        print(f"     n = {n}:  |c_n| = {amp:.4e}")

    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "buckling_mode.npy"), mode)
    print()
    print("initiating mode saved to python/buckling_mode.npy "
          "(full dof vector; add a multiple of it to the flat state to seed)")


if __name__ == "__main__":
    main()
