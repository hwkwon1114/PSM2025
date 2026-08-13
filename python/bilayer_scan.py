"""
Does the bilayer flat state have a bifurcation at all?

The claim under test: in a monolayer the flat state is a critical point (the energy is
even in the out-of-plane displacement), so buckling is a genuine bifurcation with a
threshold. In a bilayer the mismatch between the two prescribed metrics produces a
spontaneous curvature, the flat state is NOT a critical point, and there is no
bifurcation to find -- the sheet simply bends from the first increment.

The comparison is controlled: both cases carry the same MEAN growth
    rate_bot = t * field * (1 + m)
    rate_top = t * field * (1 - m)
so m = 0 is the symmetric control (which should behave like the monolayer) and m > 0
isolates the effect of the mismatch alone.

The discriminator is the norm of the energy gradient restricted to the out-of-plane
block, measured after relaxing the in-plane displacement. Zero => critical point =>
a bifurcation exists. Nonzero => the flat state is not an equilibrium at all.

Run from the repo root:  python python/bilayer_scan.py
"""

import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyshell  # noqa: E402
from verify_bindings import (  # noqa: E402
    hessian_operator,
    out_of_plane_rigid_modes,
    relax_in_plane,
)

RES = 24
RADIUS = 1.0


def build_bilayer(res=RES, swelling=1.0, mismatch=0.0, E=1.0, nu=0.5, h=0.01):
    """Bilayer disk, azimuthal sin(r)/r growth split between the layers."""
    s = pyshell.BilayerShell()
    s.init_disk(radius=RADIUS, res=res)
    s.set_material(E=E, nu=nu, h=h)

    verts = s.rest_vertices()
    faces = s.faces()
    centers = verts[faces].mean(axis=1)

    fx = centers[:, 0] / RADIUS
    fy = centers[:, 1] / RADIUS
    r = np.maximum(np.hypot(fx, fy), 1e-12)

    field = np.sin(r) / r - 1.0
    theta = np.arctan2(fy, fx)
    angles = theta + 0.5 * np.pi
    zeros = np.zeros_like(field)

    s.set_ortho_growth("bottom", angles, swelling * field * (1.0 + mismatch), zeros)
    s.set_ortho_growth("top", angles, swelling * field * (1.0 - mismatch), zeros)
    return s


def probe(shell):
    """Relax in-plane, then report the out-of-plane gradient norm and lowest eigenvalue."""
    nV = shell.n_vertices
    x_base, _ = relax_in_plane(shell)
    _, g = shell.energy_and_gradient()

    idx = np.concatenate([np.arange(2 * nV, 3 * nV), np.arange(3 * nV, shell.n_dofs)])
    Q = out_of_plane_rigid_modes(shell, x_base)

    g_out = g[idx]
    g_out = g_out - Q @ (Q.T @ g_out)      # rigid components carry no information
    g_in = g[: 2 * nV]

    H = hessian_operator(shell, x_base, idx=idx, deflate=Q)
    lam = np.sort(eigsh(H, k=3, which="SA", tol=1e-9, maxiter=40000,
                        return_eigenvectors=False))
    return np.linalg.norm(g_in), np.linalg.norm(g_out), lam[0]


def main():
    print("smoke test: BilayerShell")
    s = build_bilayer(res=16, swelling=1.0, mismatch=0.5)
    print(f"   dofs = {s.n_dofs}, vertices = {s.n_vertices}, faces = {s.n_faces}")
    print(f"   energy = {s.energy():.6e}")
    ab = s.get_abars("bottom")
    at = s.get_abars("top")
    print(f"   max |abar_bot - abar_top| = {np.abs(ab - at).max():.6e}  "
          "(nonzero => the two layers really do differ)")
    print()

    for mismatch, label in [(0.0, "SYMMETRIC  (m = 0, control)"),
                            (0.5, "MISMATCHED (m = 0.5)")]:
        print("=" * 72)
        print(f"{label}")
        print("=" * 72)
        print(f"{'t':>8}  {'||g_inplane||':>14}  {'||g_outofplane||':>17}  {'lambda_0':>13}")
        print("-" * 72)

        for t in [1e-3, 2e-3, 4e-3, 1e-2, 5e-2, 0.2, 0.6, 1.0]:
            sh = build_bilayer(res=RES, swelling=t, mismatch=mismatch)
            gin, gout, lam0 = probe(sh)
            flag = ""
            if gout > 1e-9:
                flag = "  <-- not a critical point"
            elif lam0 < 0:
                flag = "  <-- saddle"
            print(f"{t:8.4f}  {gin:14.4e}  {gout:17.4e}  {lam0:+13.5e}{flag}")
        print()

    print("Reading: for the symmetric control ||g_outofplane|| should sit at solver")
    print("tolerance (the flat state IS an equilibrium, so a threshold exists and")
    print("lambda_0 changes sign). For the mismatched case a nonzero out-of-plane")
    print("gradient means the flat state is not an equilibrium at any swelling, so")
    print("there is no bifurcation to locate and nothing to seed.")


if __name__ == "__main__":
    main()
