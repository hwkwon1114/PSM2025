#!/usr/bin/env python3
"""Smoothed dome objective with the repository's post-solve connectivity gate.

Inputs are the SI outputs of ``dome_curvature.geometric_curvature`` (principal curvatures ``values``
(n,2) in 1/m, a ``valid`` mask) plus material coordinates ``uv`` in metres. Everything is evaluated on
the 160 x 240 mm central ROI; invalid faces stay in the denominators.

Objective (maximise):  J = S * (1 - gamma * f_saddle),   S = d_+ + d_-,  gamma in [0, 1]
  * d_s = q20 of the dome strength c_s = ramp(min(s k_1, s k_2)) for normal orientation s = +/-1, where
    ramp is exactly 0 for non-positive input, quadratic on (0, h) and linear beyond (C^1; invalid faces
    contribute 0). q20 is the C-infinity smoothed area-weighted 20th percentile (units 1/m): a strong dome
    over at least 80 % of the ROI. At most one orientation can be a dome over 80 % of the area, so
    S = d_+ + d_- is exactly invariant under normal reversal, is 0 for a flat state, and has no smooth-max bias.
  * f_saddle = int K_- dA / (int K_+ dA + int K_- dA) in [0, 1] with K = k_1 k_2 (exact hinge parts; 0 when
    K vanishes everywhere): the scale-free Gauss-Bonnet share of saddle flux.
  * Monotonicity by construction: every d_s is non-decreasing under scaling all curvatures by lambda > 0
    (ramp(lambda m) is non-decreasing in lambda for every m, and the smoothed quantile is monotone), f_saddle is
    exactly scale-free, and (1 - gamma f) >= 0, so J(lambda k) is non-decreasing in lambda: uniformly
    flattening a shape can never raise J. (An additive penalty cannot have this property once J < 0.)

Hard gate (not part of J): the largest edge-connected set of coherent same-sign faces meeting the
repository criteria of ``dome_curvature.coverage`` (|k|_min >= FLOOR and |k|_min >= 0.1 |k|_max, valid)
must cover >= 80 % of the ROI material area.
"""
import argparse
import json
from collections import deque
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dome_bo'))

ROI_U_HALF = 0.08   # m
ROI_V_HALF = 0.12   # m
FLOOR = 0.1         # 1/m, same as dome_curvature.coverage
ANISOTROPY = 0.1    # |k|_min >= ANISOTROPY * |k|_max
GATE = 0.8
RAMP_H = 1e-3       # 1/m, width of the quadratic part of the dome-strength ramp


def smooth_quantile(values, weights, q: float = 0.20, epsilon: float = 0.005) -> float:
    """Smooth weighted quantile: solves sum_i w_i sigmoid((Q - v_i)/eps) = q by damped Newton."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    if len(values) == 0 or weights.sum() <= 0:
        return 0.0
    norm_w = weights / weights.sum()
    order = np.argsort(values)
    idx = min(int(np.searchsorted(np.cumsum(norm_w[order]), q)), len(values) - 1)
    Q = float(values[order[idx]])
    eps = max(1e-6, float(epsilon))
    lo, hi = float(values.min()), float(values.max())

    for _ in range(40):
        z = np.clip((Q - values) / eps, -50.0, 50.0)
        sig = 1.0 / (1.0 + np.exp(-z))
        f = float(np.sum(norm_w * sig) - q)
        if abs(f) < 1e-9:
            break
        df = max(1e-12, float(np.sum(norm_w * sig * (1.0 - sig))) / eps)
        span = 0.5 * (hi - lo + eps)
        Q = float(np.clip(Q - np.clip(f / df, -span, span), lo - 2.0 * eps, hi + 2.0 * eps))
    return Q


def softplus(x, tau: float = 1e-4):
    """tau*ln(1+exp(x/tau)); exact (= x) for x/tau > 30, so large inputs never saturate."""
    x = np.asarray(x, float)
    z = x / tau
    return np.where(z > 30.0, x, tau * np.log1p(np.exp(np.clip(z, -50.0, 30.0))))


def ramp(x, h: float = RAMP_H):
    """C^1 non-negative ramp: 0 for x <= 0, x^2/(2h) on (0, h), x - h/2 beyond. Non-decreasing in x."""
    x = np.asarray(x, float)
    return np.where(x <= 0.0, 0.0, np.where(x < h, x * x / (2.0 * h), x - 0.5 * h))


def gauss_bonnet_integral(gaussian_curvature, weights, tau: float = 1e-4, k_floor: float = 0.0) -> dict:
    """Softplus-partitioned Gauss-Bonnet fluxes and the saddle share int K_- / (int |K| + k_floor * area).

    k_floor (1/m^2) regularises the 0/0 of an essentially flat state so the share is continuous at K -> 0
    (a flat state has no saddle flux); it is negligible for any real dome (K ~ 1e-2 1/m^2 and above).
    """
    K = np.asarray(gaussian_curvature, float)
    w = np.asarray(weights, float)
    pos = float(np.sum(w * softplus(K, tau)))
    neg = float(np.sum(w * softplus(-K, tau)))
    return {"integral_K_pos": pos, "integral_K_neg": neg,
            "saddle_flux_fraction": neg / max(pos + neg + k_floor * float(np.sum(w)), 1e-30)}


def face_adjacency(faces) -> list:
    """Edge-sharing face adjacency."""
    edge_faces = {}
    for f, tri in enumerate(np.asarray(faces)):
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(f)
    adj = [[] for _ in range(len(faces))]
    for shared in edge_faces.values():
        if len(shared) == 2:
            adj[shared[0]].append(shared[1])
            adj[shared[1]].append(shared[0])
    return adj


def material_areas(uv, faces):
    t = np.asarray(uv, float)[np.asarray(faces)]
    d1, d2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]
    return np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / 2.0, t.mean(axis=1)


def coherent_masks(values, valid):
    """Per-sign eligibility mirroring dome_curvature.coverage (before the ROI restriction)."""
    values = np.asarray(values, float)
    finite = np.isfinite(values).all(axis=1) & np.asarray(valid, bool)
    safe = np.where(finite[:, None], values, 0.0)
    small, large = np.min(np.abs(safe), axis=1), np.max(np.abs(safe), axis=1)
    eligible = finite & (small >= FLOOR) & (small >= ANISOTROPY * large)
    return {s: eligible & (s * safe > 0).all(axis=1) for s in (-1, 1)}


def connected_gate(masks_sign, area, roi, adj) -> dict:
    """Largest edge-connected component of `masks_sign` inside the ROI, as a fraction of ROI area."""
    member = masks_sign & roi
    seen = np.zeros(len(member), bool)
    best = 0.0
    for start in np.flatnonzero(member):
        if seen[start]:
            continue
        seen[start] = True
        queue, total = deque([start]), 0.0
        while queue:
            f = queue.popleft()
            total += area[f]
            for n in adj[f]:
                if member[n] and not seen[n]:
                    seen[n] = True
                    queue.append(n)
        best = max(best, total)
    roi_area = float(area[roi].sum())
    return {"total_fraction": float(area[member].sum() / roi_area),
            "largest_connected_fraction": float(best / roi_area)}


def evaluate_dome_objective(values, valid, uv, faces, q: float = 0.20, epsilon: float = 0.05,
                            gamma_saddle: float = 1.0, roi_half=(ROI_U_HALF, ROI_V_HALF)) -> dict:
    """Smoothed objective J and the hard connectivity gate on one solved state (SI units).

    ``roi_half`` = (u, v) half-extents in metres of the central face-centroid ROI (default: the historical
    160 x 240 mm rectangular ROI; the square-sheet stage passes (0.08, 0.08))."""
    values = np.asarray(values, float)
    valid = np.asarray(valid, bool)
    uv = np.asarray(uv, float)
    faces = np.asarray(faces, int)
    if uv.size and np.max(np.abs(uv)) > 1.0:
        raise ValueError("uv must be in metres (SI), as produced for geometric_curvature")
    area, centers = material_areas(uv, faces)
    roi = (np.abs(centers[:, 0]) <= roi_half[0]) & (np.abs(centers[:, 1]) <= roi_half[1])
    if not roi.any():
        raise ValueError("Empty central region")
    adj = face_adjacency(faces)

    masks = coherent_masks(values, valid)
    gates = {s: connected_gate(masks[s], area, roi, adj) for s in (-1, 1)}
    sign = max((-1, 1), key=lambda s: (gates[s]["largest_connected_fraction"], gates[s]["total_fraction"]))
    gate = gates[sign]

    finite = np.isfinite(values).all(axis=1) & valid
    safe = np.where(finite[:, None], values, 0.0)
    K = np.where(finite, np.prod(safe, axis=1), 0.0)
    pos = float(np.sum(area[roi] * np.maximum(K[roi], 0.0)))
    neg = float(np.sum(area[roi] * np.maximum(-K[roi], 0.0)))
    gb = {"integral_K_pos": pos, "integral_K_neg": neg,
          "saddle_flux_fraction": neg / (pos + neg) if pos + neg > 0.0 else 0.0}

    # Dome strength per normal orientation; at most one is non-zero (a >= 80 % dome needs one sign).
    strengths = []
    for s_ in (-1, 1):
        c_s = np.where(finite, ramp((s_ * safe).min(axis=1)), 0.0)
        strengths.append(max(0.0, smooth_quantile(c_s[roi], area[roi], q=q, epsilon=epsilon)))   # smoothing can undershoot 0
    j_quantile = float(sum(strengths))
    gamma = float(np.clip(gamma_saddle, 0.0, 1.0))
    j_combined = j_quantile * (1.0 - gamma * gb["saddle_flux_fraction"])
    roi_area = float(area[roi].sum())
    return {
        "dominant_sign": int(sign),               # gate orientation only; J does not use it
        "dome_strength": j_quantile,
        "j_quantile": j_quantile,
        "saddle_flux_fraction": gb["saddle_flux_fraction"],
        "j_combined": j_combined,
        "mean_K_pos": gb["integral_K_pos"] / roi_area,
        "mean_K_neg": gb["integral_K_neg"] / roi_area,
        "total_coherent_fraction": gate["total_fraction"],
        "largest_connected_fraction": gate["largest_connected_fraction"],
        "valid_area_fraction": float(area[roi & valid].sum() / roi_area),
        "feasibility_margin": gate["largest_connected_fraction"] - GATE,
        "qualified": bool(gate["largest_connected_fraction"] >= GATE),
    }


def evaluate_state(state: dict, radius: float = 0.015, roi_half=(ROI_U_HALF, ROI_V_HALF)) -> dict:
    """Curvature + objective for a native_hybrid state dict (positions/material_uv in metres)."""
    from dome_curvature import geometric_curvature
    positions = np.asarray(state["positions"], float)
    uv = np.asarray(state["material_uv"], float)
    faces = np.asarray(state["faces"], int)
    values, _, valid = geometric_curvature(positions, uv, faces, radius=radius)
    return evaluate_dome_objective(values, valid, uv, faces, roi_half=roi_half)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the dome objective and connectivity gate")
    parser.add_argument("--state", type=Path, required=True, help="native_hybrid_state.json")
    args = parser.parse_args()
    for k, v in evaluate_state(json.loads(args.state.read_text())).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
