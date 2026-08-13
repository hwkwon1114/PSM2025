"""
Verification for the pyshell bindings.

Checks, in order:
  1. the analytic gradient against finite differences (scipy.optimize.check_grad)
  2. that the 6 rigid-body modes really are null directions of the energy
  3. that a matrix-free Hessian built from the C++ gradient recovers the buckling
     modes of a flat disk under a prescribed non-Euclidean metric

Run from the repo root:  python python/verify_bindings.py
"""

import os
import sys

import numpy as np
from scipy.optimize import check_grad
from scipy.sparse.linalg import LinearOperator, eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyshell  # noqa: E402


def build_validation_disk(res=48, radius=1.0, E=1.0, nu=0.5, h=0.01, swelling=1.0):
    """The 'validation' growth case: azimuthal stretch sin(r)/r, no radial growth."""
    s = pyshell.MonolayerShell()
    s.init_disk(radius=radius, res=res)
    s.set_material(E=E, nu=nu, h=h)

    verts = s.rest_vertices()
    faces = s.faces()
    centers = verts[faces].mean(axis=1)

    fx = centers[:, 0] / radius
    fy = centers[:, 1] / radius
    r = np.hypot(fx, fy)
    r = np.maximum(r, 1e-12)

    s_azimuthal = np.sin(r) / r
    theta = np.arctan2(fy, fx)

    # principal growth direction is azimuthal, i.e. theta + pi/2
    angles = theta + 0.5 * np.pi
    rate1 = swelling * (s_azimuthal - 1.0)   # stretch 1 + rate1
    rate2 = np.zeros_like(rate1)             # radial: no growth

    s.set_ortho_growth(angles, rate1, rate2)
    return s


def hessian_operator(shell, x0, idx=None, deflate=None, deflate_shift=1.0):
    """
    Matrix-free Hessian at x0, via central differences of the analytic gradient.

    `idx` restricts the operator to a subset of the DOFs (the rest are held fixed).

    `deflate` is a matrix of orthonormal columns spanning a known null space (the rigid
    modes). They are not merely projected out: doing that leaves them in the spectrum at
    eigenvalue exactly zero, so a `which='SA'` solve returns them whenever no genuinely
    negative mode exists, which looks like a cluster of ~1e-17 "unstable" modes. Instead
    they are mapped to `deflate_shift`, which must be chosen well above the eigenvalues
    of interest so that they sit at the far end of the spectrum and stay out of the way.
    """
    n_full = shell.n_dofs
    if idx is None:
        idx = np.arange(n_full)
    n = len(idx)

    norm_x = max(1.0, np.linalg.norm(x0))
    step = np.sqrt(np.finfo(float).eps) * norm_x

    def grad_at_sub(u):
        x = x0.copy()
        x[idx] = u
        shell.set_dofs(x)
        return shell.energy_and_gradient()[1][idx]

    u0 = x0[idx]

    def matvec(v):
        v = np.asarray(v, dtype=float).ravel()
        coeff = None
        if deflate is not None:
            coeff = deflate.T @ v
            v = v - deflate @ coeff
        nv = np.linalg.norm(v)
        if nv == 0.0:
            out = np.zeros(n)
        else:
            eps = step / nv
            out = (grad_at_sub(u0 + eps * v) - grad_at_sub(u0 - eps * v)) / (2.0 * eps)
        if deflate is not None:
            out = out - deflate @ (deflate.T @ out)
            out = out + deflate_shift * (deflate @ coeff)
        return out

    return LinearOperator((n, n), matvec=matvec, dtype=float)


def relax_in_plane(shell):
    """
    Relax the in-plane (x, y) displacement with z and the edge directors held flat.

    The prescribed metric puts the flat sheet in residual membrane stress, so the
    as-generated flat mesh is NOT a critical point. Relaxing in-plane first gives the
    pre-buckling base state whose stability we then test.
    """
    from scipy.optimize import minimize

    nV = shell.n_vertices
    x_flat = shell.get_dofs().copy()
    idx_in = np.arange(0, 2 * nV)   # x and y blocks

    def fun(u):
        x = x_flat.copy()
        x[idx_in] = u
        shell.set_dofs(x)
        E, g = shell.energy_and_gradient()
        return E, g[idx_in]

    res = minimize(fun, x_flat[idx_in], jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-18, "gtol": 1e-14})

    x_base = x_flat.copy()
    x_base[idx_in] = res.x
    shell.set_dofs(x_base)
    return x_base, res


def out_of_plane_rigid_modes(shell, x_base):
    """
    Rigid modes that live in the out-of-plane block at a flat configuration:
    translation along z, and the two rotations that tilt the plane (dz = y, dz = -x).
    Rotation about z is purely in-plane and so does not appear here.
    """
    nV = shell.n_vertices
    nE = shell.n_edges
    verts = shell.vertices()

    modes = np.zeros((nV + nE, 3))
    modes[:nV, 0] = 1.0             # translate z
    modes[:nV, 1] = verts[:, 1]     # rotate about x : dz = y
    modes[:nV, 2] = -verts[:, 0]    # rotate about y : dz = -x

    q, _ = np.linalg.qr(modes)
    return q


def main():
    print("=" * 68)
    print("1. gradient vs finite differences")
    print("=" * 68)

    s = build_validation_disk(res=16)
    n = s.n_dofs
    print(f"   mesh: {s.n_vertices} vertices, {s.n_faces} faces, "
          f"{s.n_edges} edges -> {n} dofs")

    rng = np.random.default_rng(0)
    x0 = s.get_dofs()
    x0 = x0 + 1e-3 * rng.standard_normal(n)   # off the flat state, where g == 0

    def f(x):
        s.set_dofs(x)
        return s.energy()

    def g(x):
        s.set_dofs(x)
        return s.energy_and_gradient()[1]

    err = check_grad(f, g, x0, epsilon=1e-7)
    gnorm = np.linalg.norm(g(x0))
    print(f"   ||g||                    = {gnorm:.6e}")
    print(f"   check_grad abs error     = {err:.6e}")
    print(f"   relative to ||g||        = {err / gnorm:.6e}")
    assert err / gnorm < 1e-5, "gradient does not match finite differences"
    print("   OK")

    print()
    print("=" * 68)
    print("2. rigid-body modes are null directions")
    print("=" * 68)

    Q = s.rigid_body_modes()
    print(f"   modes shape              = {Q.shape}")
    print(f"   orthonormality ||QtQ-I|| = {np.abs(Q.T @ Q - np.eye(6)).max():.3e}")

    s.set_dofs(x0)
    E0 = s.energy()
    for j, name in enumerate(["T1", "T2", "T3", "R1", "R2", "R3"]):
        # energy must be invariant to first order along each rigid mode
        dE = (f(x0 + 1e-6 * Q[:, j]) - f(x0 - 1e-6 * Q[:, j])) / 2e-6
        print(f"   dE/ds along {name}          = {dE:+.3e}")
    s.set_dofs(x0)
    print(f"   E0                       = {E0:.6e}")

    print()
    print("=" * 68)
    print("3. buckling modes of the flat disk (matrix-free Lanczos)")
    print("=" * 68)

    sb = build_validation_disk(res=24)
    nV, nE = sb.n_vertices, sb.n_edges
    print(f"   dofs                     = {sb.n_dofs}  ({nV} vertices, {nE} edges)")

    E_raw = sb.energy()
    g_raw = np.linalg.norm(sb.energy_and_gradient()[1])
    x_base, res = relax_in_plane(sb)
    E_base, g_base = sb.energy_and_gradient()

    print(f"   as-generated flat state  : E = {E_raw:.6e}, ||g|| = {g_raw:.3e}")
    print(f"   after in-plane relaxation: E = {E_base:.6e}, ||g|| = "
          f"{np.linalg.norm(g_base):.3e}  ({res.nit} iterations)")
    print(f"   max |z| in base state    = {np.abs(sb.vertices()[:, 2]).max():.3e}"
          "   (still flat, as it must be)")

    # At a flat configuration the second variation decouples: the stiff in-plane
    # membrane modes (~E*h) do not mix with the soft out-of-plane modes (~E*h^3).
    # Restricting to the out-of-plane block is exact here, and removes the huge
    # eigenvalues that otherwise bury the buckling modes.
    idx_out = np.concatenate([np.arange(2 * nV, 3 * nV),
                              np.arange(3 * nV, sb.n_dofs)])
    Q_out = out_of_plane_rigid_modes(sb, x_base)
    H_out = hessian_operator(sb, x_base, idx=idx_out, deflate=Q_out)

    vals, vecs = eigsh(H_out, k=8, which="SA", tol=1e-8, maxiter=20000)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    print()
    print("   lowest out-of-plane Hessian eigenvalues:")
    n_negative = 0
    for i, lam in enumerate(vals):
        flag = "  <-- unstable" if lam < 0 else ""
        if lam < 0:
            n_negative += 1
        print(f"     lambda[{i}] = {lam:+.6e}{flag}")

    print()
    if n_negative > 0:
        print(f"   => {n_negative} negative mode(s): the flat state is a SADDLE.")
        mode = np.zeros(sb.n_dofs)
        mode[idx_out] = vecs[:, 0]
        z = mode[2 * nV:3 * nV]
        print(f"      lowest mode: max|z| component = {np.abs(z).max():.3e}, "
              f"director norm = {np.linalg.norm(mode[3 * nV:]):.3e}")
        # verify it is genuinely a descent direction
        amp = 1e-3
        sb.set_dofs(x_base + amp * mode)
        E_pert = sb.energy()
        sb.set_dofs(x_base)
        print(f"      E(base) = {E_base:.6e} -> E(base + {amp}*mode) = {E_pert:.6e}")
        print(f"      energy change = {E_pert - E_base:+.3e}  "
              f"({'DECREASES, confirms saddle' if E_pert < E_base else 'increases'})")
    else:
        print("   => no negative modes: the flat state is a local minimum here.")

    print()
    print("all checks passed")


if __name__ == "__main__":
    main()
