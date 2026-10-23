"""
Acceptance gates for the two TinyAD correctness fixes in pyshell.cpp:

  FIX #2  public energy_and_gradient_tinyad() must be FINITE (no NaN) at an exactly-flat
          (coplanar) state -- the smooth signed dihedral theta = atan2(sinComp, cosComp).
  FIX #1  Dirichlet masking: TinyAD gradient/Hessian must zero constrained DOFs exactly as
          production energy_and_gradient() does (fixed vertices -> grad 0, Hessian row/col 0
          with unit diagonal).

Gates (mono AND bilayer):
  1. NO REGRESSION  : at a curved state, TinyAD grad == production grad (~1e-14);
                      hessian symmetry and sparse==HvP (~1e-15).
  2. FLAT FINITENESS : at the flat rest state, TinyAD grad has 0 non-finite entries and
                       hessian_tinyad() is finite.
  3. FIXED BOUNDARY  : on a fixed_boundary disk, TinyAD grad on constrained DOFs ~ 0 (== prod),
                       and Hessian rows on constrained DOFs are just the unit diagonal.

Run from python/:  python verify_tinyad_fixes.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyshell  # noqa: E402
from branch_continuation import build_bilayer_onesided  # noqa: E402

RADIUS = 1.0
results = []  # (name, pass_bool, detail)


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


# ---------------------------------------------------------------------------
def _azimuthal_growth(s, swelling):
    verts = s.rest_vertices()
    faces = s.faces()
    centers = verts[faces].mean(axis=1)
    fx, fy = centers[:, 0] / RADIUS, centers[:, 1] / RADIUS
    r = np.maximum(np.hypot(fx, fy), 1e-12)
    angles = np.arctan2(fy, fx) + 0.5 * np.pi
    rate1 = swelling * (np.sin(r) / r - 1.0)
    rate2 = np.zeros_like(rate1)
    return angles, rate1, rate2


def build_mono_disk(res=16, fixed_boundary=False, swelling=1.0):
    s = pyshell.MonolayerShell()
    s.init_disk(radius=RADIUS, res=res, fixed_boundary=fixed_boundary)
    s.set_material(E=1.0, nu=0.5, h=0.01)
    a, r1, r2 = _azimuthal_growth(s, swelling)
    s.set_ortho_growth(a, r1, r2)
    return s


def build_bilayer_disk(res=16, fixed_boundary=False, t=0.5):
    s = pyshell.BilayerShell()
    s.init_disk(radius=RADIUS, res=res, fixed_boundary=fixed_boundary)
    s.set_material(E=1.0, nu=0.5, h=0.01)
    verts = s.rest_vertices()
    faces = s.faces()
    centers = verts[faces].mean(axis=1)
    fx, fy = centers[:, 0] / RADIUS, centers[:, 1] / RADIUS
    r = np.maximum(np.hypot(fx, fy), 1e-12)
    field = np.sin(r) / r - 1.0
    angles = np.arctan2(fy, fx) + 0.5 * np.pi
    zeros = np.zeros_like(field)
    s.set_ortho_growth("bottom", angles, t * field, zeros)
    s.set_ortho_growth("top", angles, zeros, zeros)
    return s


def curved_perturb(s, seed=1, amp=5e-2):
    nV, nE = s.n_vertices, s.n_edges
    x0 = s.get_dofs().copy()
    rng = np.random.default_rng(seed)
    x0[2 * nV:3 * nV] += amp * rng.standard_normal(nV)
    x0[3 * nV:] += amp * rng.standard_normal(nE)
    x0[:2 * nV] += 0.1 * amp * rng.standard_normal(2 * nV)
    s.set_dofs(x0)
    return x0


def fixed_dof_indices(s):
    """Constrained DOFs for a fixed_boundary disk: rim vertices (all 3 comps). Mirrors the
    C++ CircularPlate condition |x^2+y^2 - R^2| < 1e-6*R, so it is an INDEPENDENT check."""
    nV = s.n_vertices
    verts = s.rest_vertices()
    r2 = verts[:, 0] ** 2 + verts[:, 1] ** 2
    fixed_v = np.where(np.abs(r2 - RADIUS ** 2) < 1e-6 * RADIUS)[0]
    return fixed_v, np.concatenate([fixed_v, nV + fixed_v, 2 * nV + fixed_v])


# ---------------------------------------------------------------------------
def gate1(s, label):
    print(f"\n-- Gate 1 NO-REGRESSION [{label}] --")
    curved_perturb(s, seed=1)
    zmax = np.abs(s.vertices()[:, 2]).max()
    E_ad, g_ad = s.energy_and_gradient_tinyad()
    E_pr, g_pr = s.energy_and_gradient()
    g_ad = np.asarray(g_ad, float)
    g_pr = np.asarray(g_pr, float)
    e_rel = abs(E_ad - E_pr) / max(abs(E_pr), 1e-300)
    g_rel = np.abs(g_ad - g_pr).max() / max(np.abs(g_pr).max(), 1e-300)
    record(f"{label} energy match", e_rel < 1e-10, f"rel={e_rel:.3e} (max|z|={zmax:.2e})")
    record(f"{label} gradient match", g_rel < 1e-9, f"max-rel={g_rel:.3e}")

    H = s.hessian_tinyad()
    Hd = H.toarray()
    sym = np.abs(Hd - Hd.T).max() / max(np.abs(Hd).max(), 1e-300)
    record(f"{label} hessian symmetry", sym < 1e-12, f"rel-asym={sym:.3e}")

    rng = np.random.default_rng(7)
    v = rng.standard_normal(s.n_dofs)
    hv_sp = np.asarray(H @ v).ravel()
    hv_hp = np.asarray(s.hessian_vector_product_tinyad(v), float)
    hv_rel = np.abs(hv_sp - hv_hp).max() / max(np.abs(hv_hp).max(), 1e-300)
    record(f"{label} sparse==HvP", hv_rel < 1e-12, f"max-rel={hv_rel:.3e}")


def gate2(build_fn, label):
    print(f"\n-- Gate 2 FLAT-STATE FINITENESS [{label}] --")
    s = build_fn()                     # fresh build => flat coplanar rest state
    zmax = np.abs(s.vertices()[:, 2]).max()
    dmax = np.abs(s.get_dofs()[3 * s.n_vertices:]).max()
    E_ad, g_ad = s.energy_and_gradient_tinyad()
    g_ad = np.asarray(g_ad, float)
    n_nan = int(np.sum(~np.isfinite(g_ad)))
    record(f"{label} flat grad finite", n_nan == 0,
           f"{n_nan} non-finite entries (max|z|={zmax:.1e}, max|dir|={dmax:.1e}), "
           f"E_finite={np.isfinite(E_ad)}")
    H = s.hessian_tinyad()
    hfin = bool(np.all(np.isfinite(H.toarray())))
    record(f"{label} flat hessian finite", hfin, f"all-finite={hfin}")


def gate3(build_fn, label):
    print(f"\n-- Gate 3 FIXED-BOUNDARY MASKING [{label}] --")
    s = build_fn(fixed_boundary=True)
    fixed_v, cidx = fixed_dof_indices(s)
    print(f"  {len(fixed_v)} fixed rim vertices -> {len(cidx)} constrained vertex DOFs "
          f"(of {s.n_dofs})")
    curved_perturb(s, seed=2)

    _, g_ad = s.energy_and_gradient_tinyad()
    _, g_pr = s.energy_and_gradient()
    g_ad = np.asarray(g_ad, float)
    g_pr = np.asarray(g_pr, float)
    prod_fixed_max = np.abs(g_pr[cidx]).max()
    ad_fixed_max = np.abs(g_ad[cidx]).max()
    ad_free_max = np.abs(g_ad[np.setdiff1d(np.arange(s.n_dofs), cidx)]).max()
    print(f"  production fixed-DOF grad max = {prod_fixed_max:.3e}")
    print(f"  TinyAD     fixed-DOF grad max = {ad_fixed_max:.3e}")
    print(f"  TinyAD     free-DOF  grad max = {ad_free_max:.3e} (sanity: nonzero)")
    record(f"{label} grad fixed-DOF==0", ad_fixed_max <= max(prod_fixed_max, 1e-12) + 1e-14,
           f"prod={prod_fixed_max:.2e}, tinyad={ad_fixed_max:.2e}")

    H = s.hessian_tinyad().tocsr()
    worst_off = 0.0
    worst_diag_err = 0.0
    cset = set(int(i) for i in cidx)
    for i in cidx:
        row = H.getrow(int(i)).toarray().ravel()
        diag = row[int(i)]
        worst_diag_err = max(worst_diag_err, abs(diag - 1.0))
        off = row.copy()
        off[int(i)] = 0.0
        worst_off = max(worst_off, np.abs(off).max())
    record(f"{label} hessian rows masked", worst_off < 1e-14 and worst_diag_err < 1e-14,
           f"max|offdiag|={worst_off:.2e}, max|diag-1|={worst_diag_err:.2e}")


# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TinyAD FIX verification (FIX #1 masking, FIX #2 flat-state finiteness)")
    print("=" * 70)

    # Gate 1: no regression at a curved state
    gate1(build_mono_disk(res=16, swelling=1.0), "mono")
    gate1(build_bilayer_disk(res=16, t=0.5), "bilayer")

    # Gate 2: flat-state finiteness
    gate2(lambda: build_mono_disk(res=16, swelling=1.0), "mono")
    gate2(lambda: build_bilayer_disk(res=16, t=0.5), "bilayer")

    # Gate 3: fixed-boundary masking
    gate3(lambda fixed_boundary=True: build_mono_disk(res=16, fixed_boundary=fixed_boundary,
                                                      swelling=1.0), "mono")
    gate3(lambda fixed_boundary=True: build_bilayer_disk(res=16, fixed_boundary=fixed_boundary,
                                                        t=0.5), "bilayer")

    print("\n" + "=" * 70)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    for name, ok, detail in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 70)
    if n_fail == 0:
        print("TINYAD_FIXES_VERIFY: ALL PASS")
    else:
        print(f"TINYAD_FIXES_VERIFY: {n_fail} FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
