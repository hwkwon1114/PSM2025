"""
Phase 3 : global search over the multistable regime.

The high-swelling one-sided bilayer is multistable -- several distinct stable shapes
coexist at the same swelling, and plain continuation reaches whichever basin the
warm-start path falls into (Phase 2). To find the lowest-energy shape, this module
enumerates the distinct certified minima at a given swelling by multi-start:

  * seed relaxations along the lowest Hessian eigenmodes of the flat state (both signs, a
    few amplitudes), plus rigid-projected random out-of-plane perturbations and the flat
    state itself;
  * relax each seed, certify it with the second-order gate (stage_check.assess_stage);
  * dedupe the certified minima by ENERGY (rigid-invariant) and rank them by energy.

Multi-start rather than Farrell deflation on purpose : deflation needs a Newton/Jacobian
solve, and the matrix-free Hessian here is too noisy for that (see the Phase 2 note in
branch_continuation.py). Multi-start uses only relaxations and the robust shifted
eigensolver, so it is not defeated by Hessian-vector-product noise.

Two honest limitations:
  * The eigenmode seeds are heuristic soft-direction perturbations of the flat state, not
    exact bifurcation modes -- the flat state is not a critical point for the bilayer.
    Endpoint certification keeps every accepted result valid; the seeds only affect
    coverage.
  * This is a SAMPLING method. It reports the lowest-energy certified minimum *found among
    the sampled seeds*, not a proven global minimum. Coverage scales with
    n_modes/amps/n_random.

Run (submit as a job, not on a login node):
    python python/global_search.py bilayer 0.4
    python python/global_search.py monolayer 0.3
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bindings import build_validation_disk                       # noqa: E402
from branch_continuation import build_bilayer_onesided, minimize_energy  # noqa: E402
from stage_check import assess_stage, hessian_lowest_eig                 # noqa: E402


@dataclass
class Minimum:
    energy: float
    max_z: float        #: raw max|z| (reported only; NOT used for deduping -- see z_rms note)
    z_rms: float        #: rms of mean-centered z, a rigid-translation-invariant amplitude
    lam_min: float
    seed: str           #: which seed reached it
    x: np.ndarray


@dataclass
class SearchResult:
    minima: list = field(default_factory=list)   #: distinct certified, energy-sorted (0 = best found)
    tally: dict = field(default_factory=dict)    #: per-seed outcome counts
    n_seeds: int = 0


def _eigen_seeds(shell, x_flat, n_modes, amps, rng_seed):
    """
    Perturbations along the lowest out-of-plane Hessian eigenmodes, both signs.

    These are heuristic soft-direction seeds at the (generally non-critical) flat state,
    not exact bifurcation modes. A deterministic Lanczos start vector is supplied so the
    mode set is reproducible for a given rng_seed.
    """
    if n_modes < 1:
        return []
    nV = shell.n_vertices
    dim_oop = nV + shell.n_edges                       # out-of-plane block : z (nV) + directors (nE)
    v0 = np.random.default_rng(rng_seed).standard_normal(dim_oop)
    _, vecs, idx = hessian_lowest_eig(shell, x_flat, k=n_modes, mode="out_of_plane",
                                      want_vectors=True, v0=v0)
    seeds = []
    for i in range(vecs.shape[1]):
        mode = np.zeros(shell.n_dofs)
        mode[idx] = vecs[:, i]
        zmax = float(np.abs(mode[2 * nV:3 * nV]).max())     # unit physical max|z|
        mode /= (zmax if zmax > 1e-12 else float(np.linalg.norm(mode)))
        for sign in (+1, -1):
            for a in amps:
                seeds.append((f"mode{i}{'+' if sign > 0 else '-'}@{a:.2g}",
                              x_flat + sign * a * mode))
    return seeds


def enumerate_minima(build, t, E, h, n_modes=4, amps=None, n_random=3, rng_seed=0,
                     dedupe_rtol=1e-3, grad_tol_nd=1e-4, rel_curv=1e-6, verbose=True):
    """
    Enumerate the distinct certified minima of the shell at swelling ``t`` by multi-start.

    Returns a :class:`SearchResult`; ``minima`` is sorted by energy (element 0 is the
    lowest-energy shape *found among the sampled seeds* -- not a proven global minimum).
    ``amps`` are physical max|z| seed amplitudes (default (10*h, 30*h)).

    Deduping is on ENERGY (rigid-invariant): distinct energy => distinct shape. Two genuinely
    distinct but energy-degenerate branches (e.g. mirror images of a symmetric design) would
    merge -- acceptable here since the one-sided swell breaks that symmetry.
    """
    if amps is None:
        amps = (10.0 * h, 30.0 * h)
    if len(amps) == 0:
        raise ValueError("amps must be non-empty")
    if n_modes < 0 or n_random < 0:
        raise ValueError("n_modes and n_random must be non-negative")

    shell = build(t)
    nV = shell.n_vertices
    x_flat = shell.get_dofs().copy()
    shell.set_dofs(x_flat)

    tally = dict(eigen_seed_fail=0, relax_fail=0, first_order_reject=0,
                 curv_reject=0, spectral_fail=0, certified=0)

    seeds = [("flat", x_flat)]
    try:
        seeds += _eigen_seeds(shell, x_flat, n_modes, amps, rng_seed)
    except (ArpackNoConvergence, ArpackError, RuntimeError) as err:
        tally["eigen_seed_fail"] += 1
        if verbose:
            print(f"  (eigen-seed construction failed: {type(err).__name__}; "
                  "continuing with flat + random seeds)")

    if n_random > 0:
        shell.set_dofs(x_flat)
        Q = shell.rigid_body_modes()                   # rigid basis at the flat state
        rng = np.random.default_rng(rng_seed)
        for j in range(n_random):
            pert = np.zeros(shell.n_dofs)
            pert[2 * nV:3 * nV] = rng.standard_normal(nV)
            pert = pert - Q @ (Q.T @ pert)             # strip rigid components (no invisible offsets)
            zmax = float(np.abs(pert[2 * nV:3 * nV]).max())
            if zmax > 0:
                pert /= zmax
            seeds.append((f"rand{j}", x_flat + amps[-1] * pert))

    found = []
    for name, s in seeds:
        try:
            x, _ = minimize_energy(shell, s)
        except Exception as err:                       # one bad seed must not kill the search
            tally["relax_fail"] += 1
            if verbose:
                print(f"  seed {name:12s} -> relaxation failed ({type(err).__name__})")
            continue
        try:
            v = assess_stage(shell, x, E=E, h=h, grad_tol_nd=grad_tol_nd,
                             rel_curv=rel_curv, mode="full")
        except (ArpackNoConvergence, ArpackError, RuntimeError):
            tally["spectral_fail"] += 1
            continue
        if not v.first_order_ok:
            tally["first_order_reject"] += 1
            continue
        if not v.second_order_ok:
            tally["curv_reject"] += 1
            continue

        tally["certified"] += 1
        shell.set_dofs(x)
        z = shell.vertices()[:, 2]
        mz = float(np.abs(z).max())
        zr = float(np.sqrt(np.mean((z - z.mean()) ** 2)))
        Ev = float(shell.energy())
        found.append(Minimum(energy=Ev, max_z=mz, z_rms=zr, lam_min=float(v.lam_min),
                             seed=name, x=x.copy()))
        if verbose:
            print(f"  seed {name:12s} -> E={Ev:.6e}  max|z|={mz:.3e}  "
                  f"lam_min={v.lam_min:+.3e}  MIN")

    found.sort(key=lambda m: m.energy)
    distinct = []
    for m in found:
        if any(abs(m.energy - d.energy) <= dedupe_rtol * max(abs(d.energy), 1e-30)
               for d in distinct):
            continue
        distinct.append(m)
    return SearchResult(minima=distinct, tally=tally, n_seeds=len(seeds))


# ----------------------------------------------------------------------------------
# demonstration
# ----------------------------------------------------------------------------------

def _demo(case, t):
    E, nu, h, res = 1.0, 0.5, 0.01, 12
    if case == "bilayer":
        build = lambda tt: build_bilayer_onesided(tt, res=res, E=E, nu=nu, h=h)
    else:
        build = lambda tt: build_validation_disk(res=res, E=E, nu=nu, h=h, swelling=tt)

    print("=" * 72)
    print(f"Phase 3 : global search (multi-start) -- {case}, t={t}, res={res}")
    print("=" * 72)
    result = enumerate_minima(build, t, E=E, h=h)
    tal = result.tally

    print()
    print(f"   seeds={result.n_seeds}  certified={tal['certified']}  "
          f"rejects[1st/curv]={tal['first_order_reject']}/{tal['curv_reject']}  "
          f"fails[relax/spectral/eigen]="
          f"{tal['relax_fail']}/{tal['spectral_fail']}/{tal['eigen_seed_fail']}")

    minima = result.minima
    if not minima:
        print("=> no certified minimum among sampled seeds -- widen amps / n_modes / n_random")
        return
    print(f"=> {len(minima)} distinct certified minima, ranked by energy "
          "(lowest is BEST FOUND, not a proven global):")
    for i, m in enumerate(minima):
        tag = "  <== BEST FOUND" if i == 0 else ""
        print(f"   [{i}] E={m.energy:.6e}  max|z|={m.max_z:.3e}  z_rms={m.z_rms:.3e}  "
              f"lam_min={m.lam_min:+.3e}  via {m.seed}{tag}")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"best_found_{case}_t{t}.npy"), minima[0].x)
    print(f"   best-found DOFs saved to python/best_found_{case}_t{t}.npy")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "bilayer"
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    if case not in ("bilayer", "monolayer"):
        print(f"unknown case {case!r}; use 'bilayer' or 'monolayer'")
        sys.exit(2)
    _demo(case, t)
