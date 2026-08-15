"""
ACCEPTANCE GATE for the BILAYER exact TinyAD Hessian (one-sided-swell english-wheel case).

Uses branch_continuation.build_bilayer_onesided (bottom grows, top untouched), set to a
NON-flat curved state so stretching, bending AND the mixed coupling are all active.

  1. ENERGY:   TinyAD total energy == shell.energy()                    (rel 1e-8)
  2. GRADIENT: TinyAD full gradient == energy_and_gradient()[1]         (rel ~1e-6 max-norm),
               reported split vertex-block vs director-block.
  3. HvP symmetry (~1e-14) and sparse hessian_tinyad() @ v == HvP       (~1e-12).
"""
import numpy as np
import pyshell
import branch_continuation as bc

np.random.seed(0)

def main():
    t, res = 0.5, 16
    shell = bc.build_bilayer_onesided(t=t, res=res)

    nV = shell.n_vertices
    nE = shell.n_edges
    nD = shell.n_dofs
    assert nD == 3 * nV + nE

    # --- set a NON-flat curved state: add noise to z and to the director block ---
    x = shell.get_dofs().copy()
    x[2 * nV:3 * nV] += 5e-2 * np.random.randn(nV)   # z block
    x[3 * nV:] += 5e-2 * np.random.randn(nE)          # directors
    shell.set_dofs(x)

    # ---------- 1. ENERGY ----------
    E_prod = shell.energy()
    E_ad, g_ad = shell.energy_and_gradient_tinyad()
    rel_E = abs(E_ad - E_prod) / max(abs(E_prod), 1e-300)
    pass_E = rel_E < 1e-8
    print(f"[1] ENERGY   prod={E_prod:.12e}  tinyad={E_ad:.12e}  rel={rel_E:.3e}  "
          f"{'PASS' if pass_E else 'FAIL'}")

    # ---------- 2. GRADIENT ----------
    _, g_prod = shell.energy_and_gradient()
    g_ad = np.asarray(g_ad).ravel()
    g_prod = np.asarray(g_prod).ravel()

    def maxrel(a, b):
        denom = max(np.max(np.abs(b)), 1e-300)
        return np.max(np.abs(a - b)) / denom

    vslice = slice(0, 3 * nV)
    dslice = slice(3 * nV, nD)
    rel_g_vert = maxrel(g_ad[vslice], g_prod[vslice])
    rel_g_dir = maxrel(g_ad[dslice], g_prod[dslice])
    rel_g_all = maxrel(g_ad, g_prod)
    pass_g = rel_g_all < 1e-6
    print(f"[2] GRADIENT max-rel  all={rel_g_all:.3e}  vertex={rel_g_vert:.3e}  "
          f"director={rel_g_dir:.3e}  {'PASS' if pass_g else 'FAIL'}")

    # ---------- 3. HESSIAN: symmetry + sparse == HvP ----------
    v = np.random.randn(nD)
    w = np.random.randn(nD)
    Hv = np.asarray(shell.hessian_vector_product_tinyad(v)).ravel()
    Hw = np.asarray(shell.hessian_vector_product_tinyad(w)).ravel()
    sym = abs(w @ Hv - v @ Hw) / max(abs(v @ Hw), 1e-300)
    pass_sym = sym < 1e-13
    print(f"[3a] HvP symmetry |wHv - vHw|/|vHw| = {sym:.3e}  "
          f"{'PASS' if pass_sym else 'FAIL'}")

    H = shell.hessian_tinyad()
    Hv_sparse = np.asarray(H @ v).ravel()
    rel_sparse = np.max(np.abs(Hv_sparse - Hv)) / max(np.max(np.abs(Hv)), 1e-300)
    pass_sparse = rel_sparse < 1e-12
    print(f"[3b] sparse@v vs HvP max-rel = {rel_sparse:.3e}  "
          f"{'PASS' if pass_sparse else 'FAIL'}")

    allpass = pass_E and pass_g and pass_sym and pass_sparse
    print(f"\nBILAYER TINYAD ACCEPTANCE: {'ALL PASS' if allpass else 'FAIL'}")
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
