"""
Phase 1 : bifurcation-aware branch-switching continuation.

Ramp the swelling fraction, relax the shell at each stage warm-started from the previous
equilibrium, and -- whenever the relaxed state is a saddle (the Phase 0 gate fails) --
seed a perturbation along the initiating buckling eigenvector and re-minimize onto a
genuine branch. Both pitchfork signs are tried and the lower-energy branch is kept.

This replaces "grind the tolerance down until rounding noise happens to grow the unstable
mode" (fragile, mesh dependent) with a deliberate, deterministic step onto the branch the
instant it opens -- the robustness fix from docs/robust_optimization_spec.

Heavy minimization runs here in Python via scipy over the pyshell energy/gradient, with
the rigid-body nullspace projected out so L-BFGS-B stays well conditioned. The C++ HLBFGS
loop is the production home for the same logic (Phase 1 C++ in the spec).

Demo (submit as a job, not on a login node):
    python python/branch_continuation.py
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bindings import build_validation_disk  # noqa: E402
from stage_check import assess_stage, hessian_lowest_eig  # noqa: E402


# ----------------------------------------------------------------------------------
# well-conditioned energy minimization over pyshell
# ----------------------------------------------------------------------------------

def _rigid_projector(shell, x):
    """Orthonormal basis of the 6 rigid modes at configuration x (columns)."""
    shell.set_dofs(x)
    return shell.rigid_body_modes()   # (nDofs, 6), already orthonormalized


def minimize_energy(shell, x0, gtol=1e-8, ftol=1e-15, maxiter=4000):
    """
    Relax the shell to an equilibrium from x0.

    The prescribed-metric energy is invariant along the 6 rigid-body modes, so a raw
    solve wanders in that nullspace and conditions badly (this is what makes a naive
    full-DOF scipy solve crawl). We project the rigid components out of the gradient at
    each evaluation, keeping the search in the physical complement. Q is fixed at x0 for
    the stage -- exact for translations, a good approximation for rotations over one step.
    """
    Q = _rigid_projector(shell, x0)

    def fg(x):
        shell.set_dofs(x)
        E, g = shell.energy_and_gradient()
        g = g - Q @ (Q.T @ g)
        return E, g

    res = minimize(fg, np.asarray(x0, float), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol})
    return res.x, res


# ----------------------------------------------------------------------------------
# the continuation with branch switching
# ----------------------------------------------------------------------------------

@dataclass
class StageRecord:
    t: float            #: swelling fraction
    energy: float
    max_z: float        #: buckling amplitude proxy
    lam_min: float      #: smallest Hessian eigenvalue at the accepted state
    branched: int       #: 0 = stayed on the incoming branch, +/-1 = switched (sign)
    accepted: bool      #: passed the second-order gate after this stage


@dataclass
class ContinuationResult:
    x: np.ndarray = None
    records: list = field(default_factory=list)


def branch_continuation(build, schedule, E, h, eps_curv=1e-7, seed_amp=None,
                        grad_tol_nd=1e-4, verbose=True):
    """
    Parameters
    ----------
    build : callable(t) -> shell
        Builds the shell at swelling fraction t (fixed mesh/topology across t, so the DOF
        vector warm-starts cleanly from one stage to the next).
    schedule : iterable of float
        Increasing swelling fractions, e.g. the run_basic_disk ramp.
    E, h : float
        Material parameters, for the non-dimensional gradient norm.
    seed_amp : float or None
        Absolute amplitude of the (unit-norm) eigenvector perturbation used to step onto a
        branch. Defaults to 20*h. Under-seeding is the classic failure; over-seeding only
        costs a few extra solve iterations.

    Returns a :class:`ContinuationResult` with the final DOFs and a per-stage log.
    """
    if seed_amp is None:
        seed_amp = 20.0 * h

    out = ContinuationResult()
    x = None

    for t in schedule:
        shell = build(t)
        x0 = shell.get_dofs().copy() if x is None else x
        x, _ = minimize_energy(shell, x0)

        v = assess_stage(shell, x, E=E, h=h, grad_tol_nd=grad_tol_nd,
                         eps_curv=eps_curv, mode="auto")
        branched = 0

        if not v.second_order_ok:
            # saddle : step onto a branch along the initiating eigenvector, try both signs
            vals, vecs, idx = hessian_lowest_eig(shell, x, k=1, mode="auto",
                                                 want_vectors=True)
            mode = np.zeros(shell.n_dofs)
            mode[idx] = vecs[:, 0]
            nrm = np.linalg.norm(mode)
            if nrm > 0:
                mode /= nrm

            best = None
            for sign in (+1, -1):
                xs, _ = minimize_energy(shell, x + sign * seed_amp * mode)
                shell.set_dofs(xs)
                Es = shell.energy()
                if best is None or Es < best[1]:
                    best = (xs, Es, sign)
            x, _, branched = best
            v = assess_stage(shell, x, E=E, h=h, grad_tol_nd=grad_tol_nd,
                             eps_curv=eps_curv, mode="full")

        shell.set_dofs(x)
        rec = StageRecord(t=float(t), energy=float(shell.energy()),
                          max_z=float(np.abs(shell.vertices()[:, 2]).max()),
                          lam_min=float(v.lam_min), branched=int(branched),
                          accepted=bool(v.accepted))
        out.records.append(rec)
        if verbose:
            tag = "switch %+d" % branched if branched else "carry   "
            ok = "min " if v.accepted else "!MIN"
            print(f"  t={rec.t:6.3f}  E={rec.energy:.6e}  max|z|={rec.max_z:.3e}  "
                  f"lam_min={rec.lam_min:+.3e}  [{tag}] {ok}")

    out.x = x
    return out


# ----------------------------------------------------------------------------------
# demonstration
# ----------------------------------------------------------------------------------

def _demo():
    E, nu, h, res = 1.0, 0.5, 0.01, 24
    schedule = [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]

    def build(t):
        return build_validation_disk(res=res, E=E, nu=nu, h=h, swelling=t)

    print("=" * 72)
    print("Phase 1 : branch-switching continuation on the validation disk")
    print(f"   res={res}, E={E}, h={h}, {len(schedule)} swelling stages")
    print("=" * 72)

    result = branch_continuation(build, schedule, E=E, h=h)

    print()
    final = result.records[-1]
    if final.accepted and final.max_z > 1e-3:
        print(f"=> reached a buckled minimum at full swelling : "
              f"max|z|={final.max_z:.3e}, lam_min={final.lam_min:+.3e}")
        print("   (determinate branch, captured by seeding -- no tolerance grinding)")
    elif final.accepted:
        print(f"=> converged to a stable but flat state (max|z|={final.max_z:.3e}) : "
              "check the schedule crossed the bifurcation")
    else:
        print("=> final state did not pass the second-order gate -- inspect the log")

    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "branch_final_dofs.npy"), result.x)
    print("   final DOFs saved to python/branch_final_dofs.npy")


if __name__ == "__main__":
    _demo()
