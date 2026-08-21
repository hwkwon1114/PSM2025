"""
STAGE 2b acceptance gate: the FULL TinyAD energy+gradient (stretching + bending) over
ALL dofs must match the production energy() and analytic energy_and_gradient()[1].

Run from python/:  python verify_tinyad_bending.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyshell  # noqa: E402
from verify_bindings import build_validation_disk  # noqa: E402


def main():
    s = build_validation_disk(res=16, swelling=1.0)
    nV, nE, nD = s.n_vertices, s.n_edges, s.n_dofs
    print(f"mesh: {nV} vertices, {s.n_faces} faces, {nE} edges -> {nD} dofs")

    # Non-flat perturbed state so BOTH stretching and bending are active and curved.
    x0 = s.get_dofs().copy()
    rng = np.random.default_rng(1)
    seeded = False
    bmpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "buckling_mode.npy")
    if os.path.exists(bmpath):
        bm = np.load(bmpath)
        if bm.shape[0] == nD:
            x0 = x0 + 0.1 * bm / (np.linalg.norm(bm) + 1e-30) * np.sqrt(nD)
            seeded = True
            print(f"seeded z/director perturbation from buckling_mode.npy")
    # always add extra noise to z-block and director-block to guarantee curvature
    x0[2 * nV:3 * nV] += 5e-2 * rng.standard_normal(nV)   # z
    x0[3 * nV:] += 5e-2 * rng.standard_normal(nE)          # directors
    x0[:2 * nV] += 5e-3 * rng.standard_normal(2 * nV)      # small in-plane
    if not seeded:
        print("no usable buckling_mode.npy; using random perturbation")

    s.set_dofs(x0)
    print(f"max|z| = {np.abs(s.vertices()[:, 2]).max():.3e}   "
          f"director rms = {np.sqrt(np.mean(x0[3*nV:]**2)):.3e}")

    # --- energy ---
    E_prod = s.energy()
    terms = s.energy_terms()
    E_ad, g_ad = s.energy_and_gradient_tinyad()
    print()
    print("ENERGY")
    print(f"  production energy()            = {E_prod:.12e}")
    print(f"    stretching_aa               = {terms['stretching_aa']:.12e}")
    print(f"    bending_bb                  = {terms['bending_bb']:.12e}")
    print(f"  TinyAD total                  = {E_ad:.12e}")
    e_rel = abs(E_ad - E_prod) / max(abs(E_prod), 1e-300)
    print(f"  energy rel-error              = {e_rel:.3e}")
    energy_pass = e_rel < 1e-8

    # --- gradient ---
    E_prod2, g_prod = s.energy_and_gradient()
    g_ad = np.asarray(g_ad)
    g_prod = np.asarray(g_prod)

    def blockstats(name, sl):
        d = g_ad[sl] - g_prod[sl]
        scale = max(np.abs(g_prod[sl]).max(), 1e-300)
        rel = np.abs(d).max() / scale
        # location of worst entry
        j = np.argmax(np.abs(d))
        print(f"  {name:10s}: max|dg| = {np.abs(d).max():.3e}, "
              f"scale = {scale:.3e}, max-rel = {rel:.3e}  "
              f"(worst local idx {j}: ad={g_ad[sl][j]:.6e} prod={g_prod[sl][j]:.6e})")
        return rel

    print()
    print("GRADIENT (max-rel over each block)")
    vx = blockstats("x", slice(0, nV))
    vy = blockstats("y", slice(nV, 2 * nV))
    vz = blockstats("z", slice(2 * nV, 3 * nV))
    ve = blockstats("directors", slice(3 * nV, nD))
    vert_rel = max(vx, vy, vz)
    dir_rel = ve
    full_rel = np.abs(g_ad - g_prod).max() / max(np.abs(g_prod).max(), 1e-300)

    print()
    print(f"  vertex-block max-rel          = {vert_rel:.3e}")
    print(f"  director-block max-rel        = {dir_rel:.3e}")
    print(f"  full-vector max-rel           = {full_rel:.3e}")

    grad_pass = full_rel < 1e-6

    print()
    print("=" * 60)
    print(f"  ENERGY   MATCH (rel<1e-8): {'PASS' if energy_pass else 'FAIL'}  ({e_rel:.3e})")
    print(f"  GRADIENT MATCH (rel<1e-6): {'PASS' if grad_pass else 'FAIL'}  ({full_rel:.3e})")
    print(f"    vertex block : {'PASS' if vert_rel < 1e-6 else 'FAIL'}  ({vert_rel:.3e})")
    print(f"    director block: {'PASS' if dir_rel < 1e-6 else 'FAIL'}  ({dir_rel:.3e})")
    print("=" * 60)

    if energy_pass and grad_pass:
        print("STAGE2B_VERIFY: ALL PASS")
    else:
        print("STAGE2B_VERIFY: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
