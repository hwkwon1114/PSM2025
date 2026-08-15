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
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyshell  # noqa: E402
from verify_bindings import build_validation_disk  # noqa: E402
from stage_check import assess_stage, hessian_lowest_eig  # noqa: E402


def build_bilayer_onesided(t, res=24, E=1.0, nu=0.5, h=0.01, radius=1.0):
    """
    One-sided swell : grow the bottom layer, leave the top alone (the english-wheel
    configuration of the "One-Sided Swell" artifact).

    The bottom/top metric mismatch is a spontaneous curvature, so -- unlike the monolayer
    -- the flat state is NOT a critical point and the sheet bends from the first
    increment. There is no saddle to escape; the failure the artifact shows is the solver
    stopping before the (bending-scale, ~E*h^3) out-of-plane response develops. The curl
    direction is set by the mismatch sign, hence determinate on every mesh.
    """
    s = pyshell.BilayerShell()
    s.init_disk(radius=radius, res=res)
    s.set_material(E=E, nu=nu, h=h)

    verts = s.rest_vertices()
    faces = s.faces()
    centers = verts[faces].mean(axis=1)
    fx, fy = centers[:, 0] / radius, centers[:, 1] / radius
    r = np.maximum(np.hypot(fx, fy), 1e-12)

    field = np.sin(r) / r - 1.0          # azimuthal sin(r)/r growth
    angles = np.arctan2(fy, fx) + 0.5 * np.pi
    zeros = np.zeros_like(field)

    s.set_ortho_growth("bottom", angles, t * field, zeros)  # bottom swells
    s.set_ortho_growth("top", angles, zeros, zeros)         # top untouched
    return s


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


def minimize_energy_newton(shell, x0, gtol=1e-8, maxiter=2000, fd_root=1.0 / 3.0,
                           exact_hvp=None):
    """
    Curvature-aware relaxation : trust-region Newton-CG (Steihaug) with a Hessian-vector
    product, rigid modes projected out of gradient and HvP.

    Unlike L-BFGS the inner CG detects negative curvature and steps along it, so this
    descends off a saddle and across the high-swelling snap-through where the first-order
    solver stalls. It converged cleanly (`trust-ncg`) at the t=0.4 bilayer snap-through
    where plain relaxation gave up.

    The HvP is either:
      * exact -- the analytic TinyAD Hessian via shell.hessian_vector_product_tinyad (one
        pass, no differencing noise, ~half the gradient work of the FD form), or
      * finite-difference -- a central difference of the analytic gradient (2 gradient evals
        per HvP).
    exact_hvp=None (default) auto-selects exact when the shell exposes it (monolayer), else
    FD. Pass True/False to force.

    method is `trust-ncg`, NOT `trust-krylov`: trust-krylov's aggressive Lanczos model is
    tripped by the FD HvP (it stalls "bad approximation ..."); Steihaug's trust-ncg is
    robust to it. With the exact HvP either would work.
    """
    Q = _rigid_projector(shell, x0)

    def proj(v):
        v = np.asarray(v, float).ravel()
        return v - Q @ (Q.T @ v)

    def fun(x):
        shell.set_dofs(x)
        Ev, g = shell.energy_and_gradient()
        return Ev, proj(g)

    use_exact = (exact_hvp if exact_hvp is not None
                 else hasattr(shell, "hessian_vector_product_tinyad"))

    if use_exact:
        # Assemble the exact sparse Hessian ONCE per outer iterate and reuse it for all the
        # inner-CG matvecs (trust-ncg calls hessp many times at the same x). A per-matvec
        # TinyAD Hessian pass is ~40x slower than this, since one second-order pass costs far
        # more than the fast analytic-gradient evals; caching amortizes it to one assembly
        # per Newton step + cheap sparse products.
        cache = {"x": None, "H": None}

        def hessp(x, p):
            x = np.asarray(x, float)
            if cache["x"] is None or not np.array_equal(x, cache["x"]):
                shell.set_dofs(x)
                cache["H"] = shell.hessian_tinyad()      # exact sparse Hessian
                cache["x"] = x.copy()
            return proj(np.asarray(cache["H"] @ proj(p)).ravel())
    else:
        hstepc = np.finfo(float).eps ** fd_root

        def hessp(x, p):
            p = proj(p)
            pn = np.linalg.norm(p)
            if pn == 0.0:
                return np.zeros_like(p)
            hs = hstepc * max(1.0, np.linalg.norm(x)) / pn
            shell.set_dofs(x + hs * p)
            g_plus = shell.energy_and_gradient()[1]
            shell.set_dofs(x - hs * p)
            g_minus = shell.energy_and_gradient()[1]
            return proj((g_plus - g_minus) / (2.0 * hs))

    res = minimize(fun, np.asarray(x0, float), jac=True, hessp=hessp, method="trust-ncg",
                   options={"maxiter": maxiter, "gtol": gtol})
    return res.x, res


# Note : an earlier attempt used method="trust-krylov" and appeared to show that trust
# region "does not work" here. That was a false negative -- trust-krylov's aggressive model
# is tripped by the finite-difference HvP, but trust-ncg (Steihaug) above is robust to it
# and does cross the snap-through. Keep using minimize_energy_newton for curvature-aware
# work; keep minimize_energy (L-BFGS) as the cheap default for the well-behaved stages.


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
    x: np.ndarray = None              #: last CERTIFIED minimum DOF vector
    records: list = field(default_factory=list)
    stopped_at: float = None         #: swelling fraction where it stopped (None = ran to end)


def branch_continuation(build, schedule, E, h, rel_curv=1e-6, seed_amp=None,
                        seed_factors=(0.25, 1.0, 2.5), grad_tol_nd=1e-4,
                        stop_on_fail=True, minimizer=minimize_energy, verbose=True):
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
        Physical out-of-plane (max|z|) amplitude of the eigenvector seed used to step onto a
        branch; the mode is normalized to unit max|z| first, so this is mesh-independent and
        does not mix the vertex/director units of the raw eigenvector. Defaults to 20*h.
    seed_factors : sequence of float
        Multipliers on seed_amp tried in order during an adaptive escape (smallest first);
        the sweep stops at the first amplitude that reaches a certified minimum. A single
        amplitude is unreliable in the rugged high-swelling regime, so several are probed.

    Returns a :class:`ContinuationResult` with the final DOFs and a per-stage log.
    """
    if seed_amp is None:
        seed_amp = 20.0 * h

    out = ContinuationResult()
    warm = None   # last CERTIFIED state, used to warm-start the next stage

    for t in schedule:
        shell = build(t)
        x0 = shell.get_dofs().copy() if warm is None else warm
        x, _ = minimizer(shell, x0)

        # Spectral certification can fail (ARPACK non-convergence, non-finite operator).
        # Treat that as an uncertifiable stage rather than letting it abort the whole run
        # and discard the last certified result.
        spectral_error = None
        try:
            v = assess_stage(shell, x, E=E, h=h, grad_tol_nd=grad_tol_nd,
                             rel_curv=rel_curv, mode="auto")
        except (ArpackNoConvergence, ArpackError, RuntimeError) as err:
            v, spectral_error = None, type(err).__name__
        branched = 0

        # Branch-switch ONLY at a genuine first-order equilibrium that is a saddle. If the
        # gradient has not converged the state is not an equilibrium, its lowest Hessian
        # vector is not a bifurcating mode, and seeding from it would just mask a failed
        # minimization -- fall through to stop-on-fail instead.
        if v is not None and v.first_order_ok and not v.second_order_ok:
            shell.set_dofs(x)
            E_pre = float(shell.energy())
            try:
                _, vecs, idx = hessian_lowest_eig(shell, x, k=1, mode="auto",
                                                  want_vectors=True)
            except (ArpackNoConvergence, ArpackError, RuntimeError):
                vecs = None

            if vecs is not None:
                mode = np.zeros(shell.n_dofs)
                mode[idx] = vecs[:, 0]
                # normalize to unit *physical* out-of-plane amplitude, so seed_amp is a real
                # max-|z| perturbation independent of mesh resolution and of the mixed
                # vertex(length)/director(angle) units in the raw eigenvector.
                nV = shell.n_vertices
                zmax = float(np.abs(mode[2 * nV:3 * nV]).max())
                scale = zmax if zmax > 1e-12 else float(np.linalg.norm(mode))
                if scale > 0:
                    mode /= scale

                # keep the incoming state as a fallback : accept a switch only if it lands on
                # a *certified* minimum with energy no higher than where we started, so a
                # spurious trigger or runaway seed cannot replace a good state with garbage.
                # Adaptive escape : sweep several seed amplitudes x both signs and keep the
                # best certified, lower-energy candidate. A single amplitude is unreliable in
                # the rugged high-swelling region -- the same saddle escapes at one amplitude
                # but lands on another saddle at a larger one -- so we probe a range. Stop at
                # the first amplitude that yields a certified minimum on either sign (smaller
                # perturbations are preferred, staying closer to the incoming branch).
                best_x, best_v, best_E, branched = x, v, E_pre, 0
                for factor in seed_factors:
                    for sign in (+1, -1):
                        xs, _ = minimizer(shell, x + sign * factor * seed_amp * mode)
                        try:
                            vs = assess_stage(shell, xs, E=E, h=h, grad_tol_nd=grad_tol_nd,
                                              rel_curv=rel_curv, mode="full")
                        except (ArpackNoConvergence, ArpackError, RuntimeError):
                            continue
                        shell.set_dofs(xs)
                        Es = float(shell.energy())
                        if vs.accepted and Es <= best_E + 1e-12:
                            best_x, best_v, best_E, branched = xs, vs, Es, sign
                    if branched != 0:
                        break   # certified at this amplitude; do not perturb harder
                x, v = best_x, best_v

        shell.set_dofs(x)
        accepted = bool(v.accepted) if v is not None else False
        rec = StageRecord(t=float(t), energy=float(shell.energy()),
                          max_z=float(np.abs(shell.vertices()[:, 2]).max()),
                          lam_min=(float(v.lam_min) if v is not None else float("nan")),
                          branched=int(branched), accepted=accepted)
        out.records.append(rec)
        if verbose:
            tag = "switch %+d" % branched if branched else "carry   "
            if v is None:
                print(f"  t={rec.t:6.3f}  E={rec.energy:.6e}  max|z|={rec.max_z:.3e}  "
                      f"spectral check FAILED ({spectral_error})")
            else:
                ok = "min " if v.accepted else "!MIN"
                print(f"  t={rec.t:6.3f}  E={rec.energy:.6e}  max|z|={rec.max_z:.3e}  "
                      f"lam_min={rec.lam_min:+.3e}  [{tag}] {ok}")

        if accepted:
            warm = x            # only certified minima are carried forward
            out.x = x
        elif stop_on_fail:
            # Neither a plain relax nor a branch switch reached a certified minimum (or the
            # spectral check failed). Rather than propagate an uncertified state -- which
            # corrupts every later stage -- stop and report. This is the signature of a
            # genuine instability the first-order solver cannot cross (e.g. the high-swelling
            # snap-through of the one-sided bilayer), the regime that needs the Phase 2
            # curvature-aware/preconditioned solver.
            if verbose:
                why = (f"spectral check failed ({spectral_error})" if v is None else
                       f"lam_min={v.lam_min:+.3e}, ||g||_nd={v.grad_norm_nd:.2e}")
                print(f"  -> stage t={t:.3f} not certifiable ({why}); "
                      "stopping continuation at the last certified minimum.")
            out.stopped_at = float(t)
            break

    return out


# ----------------------------------------------------------------------------------
# demonstration
# ----------------------------------------------------------------------------------

def _save(result, name):
    if result.x is None:
        print("   no certified minimum -- nothing saved")
        return
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    np.save(path, result.x)
    print(f"   final DOFs saved to python/{name}")


def _demo_monolayer():
    E, nu, h, res = 1.0, 0.5, 0.01, 24
    schedule = [0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]

    def build(t):
        return build_validation_disk(res=res, E=E, nu=nu, h=h, swelling=t)

    print("=" * 72)
    print("Phase 1 : branch-switching continuation on the monolayer validation disk")
    print(f"   res={res}, E={E}, h={h}, {len(schedule)} swelling stages")
    print("   (flat state is a saddle -> expect a branch switch past threshold)")
    print("=" * 72)

    result = branch_continuation(build, schedule, E=E, h=h)

    print()
    final = result.records[-1]
    switched = any(r.branched for r in result.records)
    if final.accepted and final.max_z > 1e-3:
        print(f"=> reached a buckled minimum at full swelling : "
              f"max|z|={final.max_z:.3e}, lam_min={final.lam_min:+.3e}")
        print(f"   (branch {'captured by seeding' if switched else 'carried'} "
              "-- no tolerance grinding)")
    elif final.accepted:
        print(f"=> converged to a stable but flat state (max|z|={final.max_z:.3e}) : "
              "check the schedule crossed the bifurcation")
    else:
        print("=> final state did not pass the second-order gate -- inspect the log")
    _save(result, "branch_final_dofs.npy")


def _demo_bilayer():
    E, nu, h, res = 1.0, 0.5, 0.01, 24
    # fine steps through the low-swelling regime, where the smooth curled branch is stable
    schedule = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.7, 1.0]

    def build(t):
        return build_bilayer_onesided(t, res=res, E=E, nu=nu, h=h)

    print("=" * 72)
    print("Phase 1 : one-sided-swell bilayer (english-wheel configuration)")
    print(f"   res={res}, E={E}, h={h}, {len(schedule)} swelling stages")
    print("   (spontaneous curvature, no saddle -> a determinate curl, no switch)")
    print("=" * 72)

    result = branch_continuation(build, schedule, E=E, h=h)

    print()
    certified = [r for r in result.records if r.accepted]
    if not certified:
        print("=> no stage certified -- the solver stopped before any bending developed")
        _save(result, "bilayer_final_dofs.npy")
        return

    last = certified[-1]
    print(f"=> determinate curl, certified minima through t={last.t:.3f} : "
          f"max|z|={last.max_z:.3e}, lam_min={last.lam_min:+.3e}")
    print("   the curl develops and is certified WITHOUT cranking tol to 1e-12 -- the")
    print("   'One-Sided Swell' mechanism, fixed by the criterion + second-order gate.")
    if result.stopped_at is not None:
        print(f"   Stopped at t={result.stopped_at:.3f}: the one-sided disk hits a genuine")
        print("   secondary instability (snap-through) that the first-order scipy solver")
        print("   cannot cross. Pushing past it needs the Phase 2 curvature-aware /")
        print("   preconditioned solver -- the ill-conditioned regime flagged in the spec.")
    _save(result, "bilayer_final_dofs.npy")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "monolayer"
    if case == "bilayer":
        _demo_bilayer()
    elif case == "monolayer":
        _demo_monolayer()
    else:
        print(f"unknown case {case!r}; use 'monolayer' or 'bilayer'")
        sys.exit(2)
