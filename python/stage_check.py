"""
Phase 0 : a second-order acceptance gate for a relaxed shell state.

A first-order solver reports "converged" whenever the gradient is small. A saddle
satisfies that too : at the flat, pre-buckled configuration the energy gradient is
essentially zero, so HLBFGS (which never forms curvature) accepts it and the panel
stays flat. Only by grinding the tolerance down until rounding noise grows the unstable
mode does it eventually roll off -- which is fragile and mesh dependent.

This module adds the missing test. A state is a genuine, stable minimum only if

    (1) the gradient is small in a *mesh-consistent* (non-dimensional) sense, AND
    (2) the smallest Hessian eigenvalue is non-negative (no descent direction left).

Both must hold. Condition (2) is what a saddle fails, and it is cheap : one smallest-
eigenvalue Lanczos solve against the matrix-free Hessian already built in
verify_bindings.hessian_operator.

Reuses, unchanged, the machinery in verify_bindings:
    hessian_operator, out_of_plane_rigid_modes, relax_in_plane, build_validation_disk.

Run the self-check from the repo root:  python python/stage_check.py
    -> relaxes the flat disk, shows the gradient is tiny (first order says "done"),
       and shows the gate rejecting it because the Hessian is indefinite.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bindings import (  # noqa: E402
    build_validation_disk,
    hessian_operator,
    out_of_plane_rigid_modes,
    relax_in_plane,
)


# ----------------------------------------------------------------------------------
# non-dimensional gradient norm
# ----------------------------------------------------------------------------------

def nondim_gradient_norm(g, E, h, n_dofs):
    """
    A mesh- and material-consistent gradient norm.

    The raw ``||g||`` handed to -gradtol is dimensional : it scales with E, h and, under
    refinement, with sqrt(N) (more residual entries to sum). A fixed threshold on it is
    therefore a *different* tightness on every mesh, which is exactly why the artifact's
    curl only appears when the tolerance is cranked "a million times harder" and why it
    is mesh sensitive.

    Reducing to a per-DOF RMS removes the sqrt(N) growth; dividing by the membrane
    stiffness E*h removes the material scale. A single threshold on the result then means
    the same thing at res 24 and res 48, and across E/h. This is the pragmatic scale from
    the spec -- calibrate the *value* once, per problem, not per mesh.
    """
    g = np.asarray(g, dtype=float).ravel()
    rms = np.linalg.norm(g) / np.sqrt(max(1, n_dofs))
    scale = E * h
    if scale <= 0.0:
        return rms
    return rms / scale


# ----------------------------------------------------------------------------------
# smallest Hessian eigenvalue (matrix-free, rigid modes deflated)
# ----------------------------------------------------------------------------------

def _is_flat(shell, tol=1e-9):
    return float(np.abs(shell.vertices()[:, 2]).max()) <= tol


def hessian_lowest_eig(shell, x, k=1, mode="auto", deflate_shift=1.0,
                       lanczos_tol=1e-8, maxiter=40000, want_vectors=False,
                       v0=None):
    """
    Smallest ``k`` Hessian eigenvalue(s) at ``x``, with the rigid-body nullspace deflated
    so it cannot masquerade as a cluster of ~0 "unstable" modes.

    mode="full"
        Full-DOF Hessian with the 6 rigid modes deflated. Correct for a general,
        possibly curved equilibrium -- use this to certify an accepted stage.
    mode="out_of_plane"
        Restrict to the z + director block and deflate the 3 out-of-plane rigid modes.
        At a *flat* configuration the second variation decouples (stiff membrane ~E*h vs
        soft bending ~E*h^3), so this is exact there and strips the large membrane
        eigenvalues that would otherwise bury the buckling modes. Not valid once curved.
    mode="auto"
        out_of_plane when the state is flat, full otherwise.

    ``v0`` seeds the Lanczos iteration (pass the previous stage's eigenvector to warm
    start along a continuation). Returns ``vals`` (sorted ascending), or ``(vals, vecs,
    idx)`` when ``want_vectors`` -- ``idx`` is the DOF subset the vectors live on.
    """
    shell.set_dofs(x)
    nV = shell.n_vertices

    if mode == "auto":
        mode = "out_of_plane" if _is_flat(shell) else "full"

    if mode == "out_of_plane":
        idx = np.concatenate([np.arange(2 * nV, 3 * nV),
                              np.arange(3 * nV, shell.n_dofs)])
        Q = out_of_plane_rigid_modes(shell, x)
    elif mode == "full":
        idx = np.arange(shell.n_dofs)
        Q = shell.rigid_body_modes()
    else:
        raise ValueError(f"unknown mode {mode!r}")

    H = hessian_operator(shell, x, idx=idx, deflate=Q, deflate_shift=deflate_shift)

    # k smallest algebraic eigenvalues; eigsh needs k < n-1
    k = min(k, H.shape[0] - 2)
    out = eigsh(H, k=k, which="SA", tol=lanczos_tol, maxiter=maxiter,
                v0=v0, return_eigenvectors=want_vectors)
    if want_vectors:
        vals, vecs = out
        order = np.argsort(vals)
        return vals[order], vecs[:, order], idx
    return np.sort(np.atleast_1d(out))


# ----------------------------------------------------------------------------------
# the acceptance gate
# ----------------------------------------------------------------------------------

@dataclass
class StageVerdict:
    """Outcome of the second-order gate for one relaxed state."""
    grad_norm: float        #: dimensional ||g|| (what the current -gradtol sees)
    grad_norm_nd: float     #: non-dimensional gradient norm (mesh/material consistent)
    lam_min: float          #: smallest Hessian eigenvalue, rigid modes deflated
    first_order_ok: bool    #: gradient small enough
    second_order_ok: bool   #: no negative curvature (not a saddle)
    accepted: bool          #: both conditions -- a genuine stable minimum
    reason: str             #: human-readable summary

    def __str__(self):
        tag = "ACCEPT" if self.accepted else "REJECT"
        return (f"[{tag}] ||g||_nd={self.grad_norm_nd:.3e}  "
                f"lam_min={self.lam_min:+.3e}  ({self.reason})")


def assess_stage(shell, x, E, h, grad_tol_nd=1e-6, eps_curv=1e-7,
                 mode="auto", v0=None):
    """
    Decide whether ``x`` is a genuine stable minimum of the shell energy.

    Parameters
    ----------
    E, h
        Young's modulus and thickness, for the non-dimensional gradient norm.
    grad_tol_nd
        Threshold on the non-dimensional gradient norm (condition 1).
    eps_curv
        Negative-curvature tolerance (condition 2) : the state is a saddle when
        ``lam_min < -eps_curv``. Absorbs the finite-difference noise in the Hessian;
        scale it to the bending stiffness (~E*h^3) for a new problem. The default is
        conservative for the E=1, h=0.01 validation case.

    Returns a :class:`StageVerdict`. This is the check to call after every continuation
    stage instead of trusting the solver's first-order "converged".
    """
    shell.set_dofs(x)
    _, g = shell.energy_and_gradient()
    gnorm = float(np.linalg.norm(g))
    gnd = nondim_gradient_norm(g, E, h, shell.n_dofs)

    lam = float(hessian_lowest_eig(shell, x, k=1, mode=mode, v0=v0)[0])

    first = gnd <= grad_tol_nd
    second = lam >= -eps_curv
    accepted = first and second

    if accepted:
        reason = "gradient small and Hessian positive : stable minimum"
    elif not second and not first:
        reason = "saddle (negative curvature) and gradient not converged"
    elif not second:
        reason = "SADDLE : negative Hessian eigenvalue -- descent direction remains"
    else:
        reason = "gradient not converged"

    return StageVerdict(
        grad_norm=gnorm, grad_norm_nd=gnd, lam_min=lam,
        first_order_ok=first, second_order_ok=second,
        accepted=accepted, reason=reason,
    )


# ----------------------------------------------------------------------------------
# self-check / demonstration
# ----------------------------------------------------------------------------------

def _demo():
    E, nu, h = 1.0, 0.5, 0.01
    res, swelling = 24, 1.0

    print("=" * 70)
    print("Phase 0 gate : the flat state passes first order but fails second order")
    print("=" * 70)

    s = build_validation_disk(res=res, E=E, nu=nu, h=h, swelling=swelling)
    x_base, rep = relax_in_plane(s)   # pre-buckling base state : flat, in-plane relaxed

    v = assess_stage(s, x_base, E=E, h=h, grad_tol_nd=1e-6, eps_curv=1e-7)
    print(f"   relaxed flat state ({rep.nit} in-plane iterations)")
    print(f"   dimensional ||g||        = {v.grad_norm:.3e}   "
          f"(a loose -gradtol accepts this)")
    print(f"   non-dimensional ||g||_nd = {v.grad_norm_nd:.3e}")
    print(f"   smallest Hessian eig     = {v.lam_min:+.3e}")
    print(f"   first-order test  : {'pass' if v.first_order_ok else 'fail'}")
    print(f"   second-order test : {'pass' if v.second_order_ok else 'FAIL'}")
    print(f"   verdict           : {v}")
    print()

    if v.first_order_ok and not v.second_order_ok:
        print("   => The gate rejects the flat saddle that the current solver reports as")
        print("      converged. This is the whole failure mode, caught by one eig solve.")
    else:
        print("   => Unexpected : re-check eps_curv / grad_tol_nd against these values.")


if __name__ == "__main__":
    _demo()
