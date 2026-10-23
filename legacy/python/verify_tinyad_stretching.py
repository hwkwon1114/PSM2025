"""STAGE 2a verification: TinyAD per-face stretching energy + gradient.

1. ENERGY MATCH: TinyAD total stretching energy == energy_terms()['stretching_aa']
   to relative 1e-8.
2. GRADIENT SELF-CONSISTENCY: TinyAD gradient (vertex block) == central finite
   difference of the TinyAD stretching ENERGY to relative 1e-6 on probed DOFs.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyshell
from verify_bindings import build_validation_disk


def main():
    rng = np.random.default_rng(0)

    s = build_validation_disk(res=16, swelling=1.0)
    nD = s.n_dofs
    nV = s.n_vertices

    # Non-flat perturbed state so stretching is active.
    x0 = s.get_dofs().copy()
    x0 = x0 + 1e-2 * rng.standard_normal(nD)
    s.set_dofs(x0)

    # --- 1. ENERGY MATCH ---
    E_ad, g_ad = s.stretching_energy_and_gradient_tinyad()
    terms = s.energy_terms()
    E_ref = terms["stretching_aa"]

    energy_rel = abs(E_ad - E_ref) / max(abs(E_ref), 1e-300)
    print(f"[energy]  tinyad = {E_ad:.15e}")
    print(f"[energy]  ref    = {E_ref:.15e}")
    print(f"[energy]  rel-error = {energy_rel:.3e}")
    energy_pass = energy_rel < 1e-8

    # --- 2. GRADIENT SELF-CONSISTENCY (central FD of the TinyAD energy) ---
    def tinyad_energy(x):
        s.set_dofs(x)
        e, _ = s.stretching_energy_and_gradient_tinyad()
        return e

    # probe a handful of vertex-block DOFs (indices 0 .. 3*nV-1)
    probe = rng.choice(3 * nV, size=8, replace=False)
    eps = 1e-6
    max_grad_rel = 0.0
    for i in probe:
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        fd = (tinyad_energy(xp) - tinyad_energy(xm)) / (2 * eps)
        an = g_ad[i]
        denom = max(abs(fd), abs(an), 1e-12)
        rel = abs(fd - an) / denom
        max_grad_rel = max(max_grad_rel, rel)
        print(f"[grad] dof {i:5d}: analytic={an:+.9e} fd={fd:+.9e} rel={rel:.3e}")

    s.set_dofs(x0)  # restore
    grad_pass = max_grad_rel < 1e-6

    print(f"[grad] max rel-error = {max_grad_rel:.3e}")
    print()
    print(f"ENERGY  MATCH: {'PASS' if energy_pass else 'FAIL'} (rel {energy_rel:.3e}, tol 1e-8)")
    print(f"GRAD SELFCONS: {'PASS' if grad_pass else 'FAIL'} (rel {max_grad_rel:.3e}, tol 1e-6)")

    ok = energy_pass and grad_pass
    print("RESULT:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
