"""
Phase 3 : global search over the multistable regime.

The high-swelling one-sided bilayer is multistable -- several distinct stable shapes
coexist at the same swelling, and plain continuation reaches whichever basin the
warm-start path falls into (Phase 2). To find the *global* (lowest-energy) shape, this
module enumerates the distinct certified minima at a given swelling by multi-start:

  * seed relaxations along the lowest Hessian eigenmodes of the flat state (both signs, a
    few amplitudes) -- these span the soft directions along which competing buckled shapes
    branch off -- plus a few random out-of-plane perturbations and the plain flat state;
  * relax each seed, certify it with the second-order gate (stage_check.assess_stage);
  * dedupe the certified minima by (energy, max|z|) and rank them by energy.

Multi-start rather than Farrell deflation on purpose : deflation needs a Newton/Jacobian
solve, and the matrix-free Hessian here is too noisy for that (see the Phase 2 note in
branch_continuation.py). Multi-start uses only relaxations and the robust shifted
eigensolver, so it is not defeated by Hessian-vector-product noise.

Run (submit as a job, not on a login node):
    python python/global_search.py bilayer 0.4
    python python/global_search.py monolayer 0.3
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence, ArpackError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_bindings import build_validation_disk                       # noqa: E402
from branch_continuation import build_bilayer_onesided, minimize_energy  # noqa: E402
from stage_check import assess_stage, hessian_lowest_eig                 # noqa: E402


@dataclass
class Minimum:
    energy: float
    max_z: float
    lam_min: float
    seed: str           #: which seed reached it
    x: np.ndarray


def _eigen_seeds(shell, x_flat, n_modes, amps):
    """Perturbations along the lowest out-of-plane Hessian eigenmodes, both signs."""
    nV = shell.n_vertices
    _, vecs, idx = hessian_lowest_eig(shell, x_flat, k=n_modes, mode="out_of_plane",
                                      want_vectors=True)
    seeds = []
    for i in range(vecs.shape[1]):
        mode = np.zeros(shell.n_dofs)
        mode[idx] = vecs[:, i]
        zmax = float(np.abs(mode[2 * nV:3 * nV]).max())      # unit physical max|z|
        mode /= (zmax if zmax > 1e-12 else float(np.linalg.norm(mode)))
        for sign in (+1, -1):
            for a in amps:
                seeds.append((f"mode{i}{'+' if sign > 0 else '-'}@{a:.2g}",
                              x_flat + sign * a * mode))
    return seeds


def enumerate_minima(build, t, E, h, n_modes=4, amps=None, n_random=3, rng_seed=0,
                     dedupe_rtol=1e-3, dedupe_ztol=1e-3, grad_tol_nd=1e-4, rel_curv=1e-6,
                     verbose=True):
    """
    Enumerate the distinct certified minima of the shell at swelling ``t`` by multi-start.

    Returns them as a list of :class:`Minimum`, sorted by energy (element 0 is the global
    minimum found). ``amps`` are physical max|z| seed amplitudes (default (10*h, 30*h)).
    """
    if amps is None:
        amps = (10.0 * h, 30.0 * h)

    shell = build(t)
    nV = shell.n_vertices
    x_flat = shell.get_dofs().copy()

    seeds = [("flat", x_flat)]
    seeds += _eigen_seeds(shell, x_flat, n_modes, amps)
    rng = np.random.default_rng(rng_seed)
    for j in range(n_random):
        pert = np.zeros(shell.n_dofs)
        pert[2 * nV:3 * nV] = rng.standard_normal(nV)
        zmax = float(np.abs(pert[2 * nV:3 * nV]).max())
        pert /= (zmax if zmax > 0 else 1.0)
        seeds.append((f"rand{j}", x_flat + amps[-1] * pert))

    found = []
    for name, s in seeds:
        x, _ = minimize_energy(shell, s)
        try:
            v = assess_stage(shell, x, E=E, h=h, grad_tol_nd=grad_tol_nd,
                             rel_curv=rel_curv, mode="full")
        except (ArpackNoConvergence, ArpackError, RuntimeError):
            continue
        if not v.accepted:
            continue
        shell.set_dofs(x)
        Ev = float(shell.energy())
        mz = float(np.abs(shell.vertices()[:, 2]).max())
        found.append(Minimum(energy=Ev, max_z=mz, lam_min=float(v.lam_min),
                             seed=name, x=x.copy()))
        if verbose:
            print(f"  seed {name:12s} -> E={Ev:.6e}  max|z|={mz:.3e}  "
                  f"lam_min={v.lam_min:+.3e}  MIN")

    # dedupe : same energy (relative) and same amplitude => same shape
    found.sort(key=lambda m: m.energy)
    distinct = []
    for m in found:
        if any(abs(m.energy - d.energy) <= dedupe_rtol * max(abs(d.energy), 1e-30)
               and abs(m.max_z - d.max_z) <= dedupe_ztol for d in distinct):
            continue
        distinct.append(m)
    return distinct


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
    minima = enumerate_minima(build, t, E=E, h=h)

    print()
    if not minima:
        print("=> no certified minimum found -- widen amps / n_modes / n_random")
        return
    print(f"=> {len(minima)} distinct certified minima (ranked by energy):")
    for i, m in enumerate(minima):
        tag = "  <== GLOBAL" if i == 0 else ""
        print(f"   [{i}] E={m.energy:.6e}  max|z|={m.max_z:.3e}  "
              f"lam_min={m.lam_min:+.3e}  via {m.seed}{tag}")
    g = minima[0]
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f"global_{case}_t{t}.npy"), g.x)
    print(f"   global-minimum DOFs saved to python/global_{case}_t{t}.npy")


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "bilayer"
    t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    if case not in ("bilayer", "monolayer"):
        print(f"unknown case {case!r}; use 'bilayer' or 'monolayer'")
        sys.exit(2)
    _demo(case, t)
