"""
STAGE 3 acceptance gates for the EXACT TinyAD Hessian bindings:
    MonolayerShell.hessian_vector_product_tinyad(v)   -> H @ v
    MonolayerShell.hessian_tinyad()                   -> sparse H (scipy csc)

Gate 1 : HvP vs central-difference of the analytic gradient (FD-limited ~1e-5..1e-6).
Gate 2 : symmetry  |v1.(H v2) - v2.(H v1)| (exact, ~1e-14).
Gate 3 : sparse @ v == HvP (~1e-12) and sparse symmetry max|H - H^T|.
Gate 4 : smallest OUT-OF-PLANE eigenvalue at the swollen, in-plane-relaxed FLAT state
         (exact Hessian via eigsh) vs the FD-Hessian value from stage_check.

Run from python/:  python verify_tinyad_hessian.py
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyshell  # noqa: E402
from verify_bindings import (  # noqa: E402
    build_validation_disk,
    relax_in_plane,
    out_of_plane_rigid_modes,
)
from stage_check import hessian_lowest_eig  # noqa: E402


def curved_state(res=16, swelling=1.0, seed=1):
    """Swollen disk pushed to a NON-flat perturbed state (both stretching+bending active)."""
    s = build_validation_disk(res=res, swelling=swelling)
    nV, nE, nD = s.n_vertices, s.n_edges, s.n_dofs
    rng = np.random.default_rng(seed)
    x0 = s.get_dofs().copy()
    x0[2 * nV:3 * nV] += 5e-2 * rng.standard_normal(nV)   # z
    x0[3 * nV:] += 5e-2 * rng.standard_normal(nE)          # directors
    x0[:2 * nV] += 5e-3 * rng.standard_normal(2 * nV)      # small in-plane
    s.set_dofs(x0)
    return s, x0


def main():
    all_pass = True

    # ---------------------------------------------------------------- setup
    s, x0 = curved_state(res=16, swelling=1.0, seed=1)
    nV, nE, nD = s.n_vertices, s.n_edges, s.n_dofs
    print(f"mesh: {nV} vertices, {s.n_faces} faces, {nE} edges -> {nD} dofs")
    print(f"curved state: max|z| = {np.abs(s.vertices()[:, 2]).max():.3e}, "
          f"director rms = {np.sqrt(np.mean(x0[3*nV:]**2)):.3e}")
    rng = np.random.default_rng(7)

    def g_at(x):
        s.set_dofs(x)
        return np.asarray(s.energy_and_gradient()[1])

    # ================================================================ GATE 1
    print()
    print("=" * 64)
    print("GATE 1 : HvP vs finite-difference of the analytic gradient")
    print("=" * 64)
    eps = 1e-6
    worst_rel = 0.0
    for t in range(5):
        v = rng.standard_normal(nD)
        v /= np.linalg.norm(v)
        s.set_dofs(x0)
        hv_exact = np.asarray(s.hessian_vector_product_tinyad(v))
        hv_fd = (g_at(x0 + eps * v) - g_at(x0 - eps * v)) / (2.0 * eps)
        rel = np.linalg.norm(hv_exact - hv_fd) / max(np.linalg.norm(hv_fd), 1e-300)
        worst_rel = max(worst_rel, rel)
        print(f"   v[{t}]: ||Hv_exact - Hv_fd|| / ||Hv_fd|| = {rel:.3e}")
    s.set_dofs(x0)
    gate1 = worst_rel < 1e-4
    all_pass &= gate1
    print(f"   worst rel error = {worst_rel:.3e}   "
          f"{'PASS' if gate1 else 'FAIL'} (FD-limited, expect ~1e-5..1e-6)")

    # ================================================================ GATE 2
    print()
    print("=" * 64)
    print("GATE 2 : symmetry of the HvP operator (exact)")
    print("=" * 64)
    worst_sym = 0.0
    for t in range(5):
        v1 = rng.standard_normal(nD)
        v2 = rng.standard_normal(nD)
        s.set_dofs(x0)
        Hv2 = np.asarray(s.hessian_vector_product_tinyad(v2))
        Hv1 = np.asarray(s.hessian_vector_product_tinyad(v1))
        num = abs(v1.dot(Hv2) - v2.dot(Hv1))
        den = np.linalg.norm(Hv1) * np.linalg.norm(v2)
        rel = num / max(den, 1e-300)
        worst_sym = max(worst_sym, rel)
        print(f"   pair[{t}]: |v1.(Hv2) - v2.(Hv1)| / (||Hv1|| ||v2||) = {rel:.3e}")
    gate2 = worst_sym < 1e-11
    all_pass &= gate2
    print(f"   worst symmetry error = {worst_sym:.3e}   "
          f"{'PASS' if gate2 else 'FAIL'} (expect ~1e-14)")

    # ================================================================ GATE 3
    print()
    print("=" * 64)
    print("GATE 3 : sparse Hessian consistency with HvP, and sparse symmetry")
    print("=" * 64)
    s.set_dofs(x0)
    H = s.hessian_tinyad()
    H = sp.csc_matrix(H)
    print(f"   sparse Hessian: shape {H.shape}, nnz = {H.nnz}")
    worst_cons = 0.0
    for t in range(5):
        v = rng.standard_normal(nD)
        hv_sparse = H @ v
        s.set_dofs(x0)
        hv_op = np.asarray(s.hessian_vector_product_tinyad(v))
        rel = np.linalg.norm(hv_sparse - hv_op) / max(np.linalg.norm(hv_op), 1e-300)
        worst_cons = max(worst_cons, rel)
        print(f"   v[{t}]: ||H@v - HvP(v)|| / ||HvP(v)|| = {rel:.3e}")
    asym = abs(H - H.T)
    max_asym = asym.max() if asym.nnz else 0.0
    scale = abs(H).max()
    rel_asym = max_asym / max(scale, 1e-300)
    print(f"   max|H - H^T| = {max_asym:.3e}   (rel to max|H| = {rel_asym:.3e})")
    gate3 = (worst_cons < 1e-10) and (rel_asym < 1e-11)
    all_pass &= gate3
    print(f"   worst consistency = {worst_cons:.3e}   "
          f"{'PASS' if gate3 else 'FAIL'} (expect ~1e-12)")

    # ================================================================ GATE 4
    print()
    print("=" * 64)
    print("GATE 4 : smallest out-of-plane eigenvalue at the FLAT relaxed state")
    print("=" * 64)
    sf = build_validation_disk(res=16, swelling=1.0)
    x_base, rep = relax_in_plane(sf)
    nVf = sf.n_vertices
    print(f"   flat relaxed ({rep.nit} iters): max|z| = "
          f"{np.abs(sf.vertices()[:, 2]).max():.3e}")

    # --- exact Hessian path: restrict to out-of-plane block, deflate out-of-plane rigids ---
    sf.set_dofs(x_base)
    H_full = sp.csc_matrix(sf.hessian_tinyad())
    idx_out = np.concatenate([np.arange(2 * nVf, 3 * nVf),
                              np.arange(3 * nVf, sf.n_dofs)])
    H_out = H_full[np.ix_(idx_out, idx_out)]
    Q = out_of_plane_rigid_modes(sf, x_base)           # (nV+nE, 3), orthonormal
    shift = 1.0
    n_out = H_out.shape[0]

    def matvec(v):
        v = np.asarray(v, float).ravel()
        c = Q.T @ v
        vp = v - Q @ c
        out = H_out @ vp
        out = out - Q @ (Q.T @ out)
        out = out + shift * (Q @ c)
        return out

    H_out_defl = LinearOperator((n_out, n_out), matvec=matvec, dtype=float)
    vals_exact = np.sort(eigsh(H_out_defl, k=6, which="SA", tol=1e-9,
                               maxiter=40000, return_eigenvectors=False))
    lam_exact = float(vals_exact[0])

    # --- FD Hessian path (existing stage_check machinery) ---
    vals_fd = hessian_lowest_eig(sf, x_base, k=6, mode="out_of_plane")
    lam_fd = float(np.atleast_1d(vals_fd)[0])

    print(f"   exact  lowest 6 eigs: "
          + ", ".join(f"{x:+.4e}" for x in vals_exact[:6]))
    print(f"   FD     lowest 6 eigs: "
          + ", ".join(f"{x:+.4e}" for x in np.atleast_1d(vals_fd)[:6]))
    print()
    print(f"   smallest eigenvalue  EXACT = {lam_exact:+.6e}")
    print(f"   smallest eigenvalue  FD    = {lam_fd:+.6e}")
    print(f"   |exact - FD|               = {abs(lam_exact - lam_fd):.3e}")
    gate4 = (lam_exact < 0.0) and (abs(lam_exact - lam_fd) <
                                   0.1 * abs(lam_fd) + 1e-9)
    all_pass &= gate4
    print(f"   exact eigenvalue negative (instability seen): "
          f"{'YES' if lam_exact < 0 else 'NO'}")
    print(f"   {'PASS' if gate4 else 'FAIL'} (expect ~-3e-3, exact ~ FD)")

    # ================================================================ SUMMARY
    print()
    print("=" * 64)
    print(f"   GATE 1 (HvP vs FD grad)        : {'PASS' if gate1 else 'FAIL'}")
    print(f"   GATE 2 (symmetry)              : {'PASS' if gate2 else 'FAIL'}")
    print(f"   GATE 3 (sparse == HvP, symm)   : {'PASS' if gate3 else 'FAIL'}")
    print(f"   GATE 4 (exact eig ~ FD eig)    : {'PASS' if gate4 else 'FAIL'}")
    print("=" * 64)
    if all_pass:
        print("STAGE3_VERIFY: ALL PASS")
    else:
        print("STAGE3_VERIFY: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
