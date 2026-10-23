#!/usr/bin/env python3
"""Nested, pitch-graded orthogonal cross-hatch toolpaths with exactly monotone edge decay.

Everything below is a property of the EMITTED recipe as the simulator applies it
(ZigZagGrowth.hpp / ZigZagSequenceGrowth.hpp / Sim_Bilayer_Growth.cpp), not of a prescreen.

Two layouts, with different guarantees (``layout=``)
----------------------------------------------------
``'stroke'`` (default) -- spatially separated, pitch-graded hatch strokes.
  Each orientation is a row of parallel strokes (a 2-strip zigzag whose strips coincide, i.e. one
  capsule-shaped stroke counted as two hits per sweep) that TILE the sheet across the stepping axis with
  no overlap and no gap. The stroke pitch p(x) = p_c (1 + (f-1)(x/R)^alpha) GROWS outward, the sweep
  repeats decay outward, and the chord extents are nested (non-increasing outward). These design-level
  monotonicities are exact and tested. The EMITTED hit field is NOT exactly non-increasing: the simulator
  paints a stroke where the point-to-segment distance <= w/2 (round caps), so where two abutting strokes
  of different length end, the longer outer stroke's cap leaves a small notch next to the shorter inner
  stroke. Along a ray from the centre the count then goes full -> 0 -> full. This is unavoidable for any
  separated capsule strokes in this simulator; ``monotonicity_defect_fraction`` measures the ROI area
  fraction affected (about 2-12 % for the tested parameter range). Boundary lines between strokes are
  also double-counted (``<=``), a measure-zero set.

``'terrace'`` -- nested centred capsules (one 2-strip operation per layer, widths = 2 x terrace radius).
  The painted set of every layer is convex and contains all inner ones, so the hit field is exactly
  non-increasing along EVERY ray from the centre (defect fraction 0, tested). Its "pitch" is the radial
  spacing between successive terrace edges, p_i = p_c (1 + (f-1) x_i^alpha), realised as strip widths;
  there are no separated strokes.

Common facts
------------
* A conventional zigzag with strip pitch p < width w cannot give a gap-free, ripple-free graded field in
  this simulator (overlapping strips each count as a hit); ``conventional_zigzag_ripple`` demonstrates it.
* Voce hardening is face-shared (A and B hits harden each other). With a UNIFORM along-strip profile and
  equal growth coefficients every hit has the same weight, so the cumulative areal growth of a face is
  g * sum_{j<H} exp(-beta j) and depends only on its total hit count H, for any ordering of the layers.
  With the calibrated ``center_peak`` profile hit weights differ along a strip and monotonicity of the
  treatment can fail even for the terrace layout; ``treatment_monotonicity_violation`` measures it.
* Dose parameters are ``total_dose`` (combined effective Voce dose at the centre, < 4.95) and
  ``dose_balance`` (share of the centre sweeps given to A): the dose of a face is shared, B starts on
  material already hardened by A.
Layouts: why the default is the nested terrace (layer-cake) emission
--------------------------------------------------------------------
The pitch-graded hatch is the staircase  a(|x|) = sum_k r_k 1[x in tile_k]  of hit levels r_k (non-increasing
outward) over tiles whose width (the hatch pitch) grows outward. Its LAYER-CAKE decomposition is
a(|x|) = sum_j (r_j - r_{j+1}) 1[|x| <= x_{j+1}]: nested centred strokes with repeats (r_j - r_{j+1}) >= 0
reproduce the identical staircase (same pitches, same levels). Every nested stroke is a centred convex set,
so along any ray from the centre its indicator never increases => the emitted hit field is exactly
non-increasing outward (``layout='terrace'``).
Emitting the tiles themselves (``layout='stroke'``, separated strokes at distinct positions) gives the same
staircase in the interior but cannot be exactly monotone with this simulator: a stroke is a CAPSULE
(distance to a segment), so adjacent tiles meet with round end caps that leave a zero-treatment notch
beside a covered tile; a ray from the centre that passes through the inner tile, the notch and then the
outer tile's cap re-enters covered area from zero (a finite-area increase). Stroke ends cannot lie outside the
sheet (Sim_Bilayer_Growth.cpp:2441 forces require_inside_material_bounds=true). The stroke layout is kept for
comparison; ``treatment_monotonicity_violation`` and the notch tests quantify its violation.

* ``decode_treatment_field`` re-implements the simulator order (toolpath order, repeats, strips,
  face-shared hit counts, ``exp(-beta * previous hits)``) and returns the cumulative growth fields.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

BETA_VOCE = 0.2231435513142097
G_TOP_DEFAULT = 0.00028835116227798787
G_BOT_DEFAULT = 1.4587994001079146e-05
ORTHO_TOP_DEFAULT = -0.3989157893255133
ORTHO_BOT_DEFAULT = 0.960306323937336

HITS_PER_SWEEP = 2          # a 2-strip zigzag has two coincident strips => two hits per centroid
SHEET_HALF_U_MM = 100.0
SHEET_HALF_V_MM = 150.0
A_E_MM = 95.0               # envelope half extent across u (also the surrogate reference length)
B_E_MAX_MM = 142.0          # maximum half extent along v
MIN_LV_MM = 1.0
TILE_GAP_MM = 1e-3          # stroke tiles are emitted this much narrower so a centroid exactly on a shared edge is
                            # never counted by both strokes (the simulator's test is distance <= half-width)


def voce_effective_dose(repeats: int, beta: float = BETA_VOCE) -> float:
    """Effective single-pass-equivalent dose after N hits with Voce hardening."""
    if repeats <= 0:
        return 0.0
    return float(np.sum(np.exp(-beta * np.arange(repeats))))


def invert_voce_dose(effective_dose: float, beta: float = BETA_VOCE) -> int:
    """Smallest integer hit count whose effective dose reaches the target."""
    limit = 1.0 / (1.0 - np.exp(-beta))
    if effective_dose >= limit - 1e-3:
        raise ValueError(f"effective dose {effective_dose} is not reachable (asymptote {limit:.3f})")
    accum, repeats = 0.0, 0
    while accum < effective_dose - 1e-6:
        accum += np.exp(-beta * repeats)
        repeats += 1
    return max(1, repeats)


# ----------------------------------------------------------------------------------------------
# Simulator-faithful decoding of an emitted operation
# ----------------------------------------------------------------------------------------------
def zigzag_segments(lv_mm: float, alpha_deg: float, n_strips: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Strip centre-line segments in the local zigzag frame (mm); mirrors buildZigZagStrips."""
    n_incl = n_strips - 2
    p = lv_mm * np.tan(np.radians(alpha_deg))
    y_bot, y_top = -0.5 * lv_mm, 0.5 * lv_mm
    x_l = -0.5 * n_incl * p
    pts = [np.array([x_l, y_bot]), np.array([x_l, y_top])]
    for j in range(1, n_incl + 1):
        pts.append(np.array([x_l + j * p, y_bot if j % 2 == 1 else y_top]))
    last = pts[-1]
    y_other = y_bot if abs(last[1] - y_top) < 1e-12 else y_top
    pts.append(np.array([x_l + n_incl * p, y_other]))
    segs = [(pts[0], pts[1])]
    segs += [(pts[k], pts[k + 1]) for k in range(1, n_incl + 1)]
    segs.append((pts[-2], pts[-1]))
    return segs


def _segment_distance_and_s(px, py, a, b):
    """Distance to a segment and the clamped path coordinate s in [0,1] (as the simulator computes)."""
    d = b - a
    denom = float(d @ d)
    if denom <= 1e-30:
        s = np.full(np.shape(px), 0.5)
    else:
        s = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / denom, 0.0, 1.0)
    return np.hypot(px - (a[0] + s * d[0]), py - (a[1] + s * d[1])), s


def _local_frame(op: Dict, U, V):
    rot = np.radians(op['rotation_deg'])
    c, s = np.cos(rot), np.sin(rot)
    cx = op['center_uv_mm'][0] + op['shift_uv_mm'][0]
    cy = op['center_uv_mm'][1] + op['shift_uv_mm'][1]
    dx, dy = np.asarray(U, float) - cx, np.asarray(V, float) - cy
    return c * dx + s * dy, -s * dx + c * dy               # inverse rotation into the zigzag frame


def operation_hits(op: Dict, U, V) -> np.ndarray:
    """Hits per material point (mm) from one operation: ONE hit per strip within half-width,
    exactly as ZigZagGrowth.hpp accumulates them (overlapping strips each count)."""
    lx, ly = _local_frame(op, U, V)
    half_w = 0.5 * op['width_mm']
    hits = np.zeros(np.shape(lx), dtype=float)
    for a, b in zigzag_segments(op['lv_mm'], op['alpha_deg'], op['n_strips']):
        hits += _segment_distance_and_s(lx, ly, a, b)[0] <= half_w
    return hits


def decode_hit_field(recipe: Dict, U, V, orientation: str = None) -> np.ndarray:
    """Total simulator hits per material point (mm) accumulated over emitted enabled toolpaths."""
    hits = np.zeros(np.shape(U), dtype=float)
    for tp in recipe['toolpaths']:
        if not tp.get('enabled', True):
            continue
        if orientation is not None and not tp['id'].startswith(f'crosshatch_{orientation}_'):
            continue
        hits += tp['repeat'] * operation_hits(tp['operation'], U, V)
    return hits


def _per_strip(value, n):
    return list(value) if isinstance(value, (list, tuple)) else [value] * n


def _profile(op: Dict, s):
    prof = op.get('profile', {'mode': 'uniform'})
    mode = prof.get('mode', 'uniform')
    if mode == 'uniform':
        return np.ones_like(s)
    if mode == 'center_peak':
        end = min(1.0, max(0.0, prof.get('top_end_ratio', 1.0)))
        power = max(1e-12, prof.get('top_profile_power', 1.0))
        return end + (1.0 - end) * np.sin(np.pi * s) ** power
    raise ValueError(f"unsupported profile mode {mode}")


def decode_treatment_field(recipe: Dict, U, V, beta: float = None) -> Dict[str, np.ndarray]:
    """Cumulative treatment at material points (mm), reproducing the simulator order exactly.

    Toolpaths in recipe order, `repeat` cycles each, strips in index order within a cycle; every hit
    uses q = exp(-beta * previous face hits) (face-shared, plastic_hits count) and adds
    q * gtop * profile * (1 +/- ortho_top) (top) and q * gbot * (1 +/- ortho_bottom) (bottom) to the two
    in-plane principal growths. Returns the hit count, the trace (g1+g2) of the top and bottom growth,
    and for each orientation the mid-plane growth ALONG and ACROSS its strips (A strips run along v).
    """
    beta = recipe['hardening']['beta'] if beta is None else beta
    shape = np.shape(U)
    count = np.zeros(shape)
    out = {'top_trace': np.zeros(shape), 'bottom_trace': np.zeros(shape)}
    for o in 'AB':
        out[f'along_{o}'] = np.zeros(shape)
        out[f'across_{o}'] = np.zeros(shape)
    for tp in recipe['toolpaths']:
        if not tp.get('enabled', True):
            continue
        op = tp['operation']
        o = tp['id'].split('_')[1] if tp['id'].startswith(('crosshatch_', 'square_')) else None
        lx, ly = _local_frame(op, U, V)
        half_w = 0.5 * op['width_mm']
        segs = zigzag_segments(op['lv_mm'], op['alpha_deg'], op['n_strips'])
        gtop, gbot = _per_strip(op['gtop'], len(segs)), _per_strip(op.get('gbot', 0.0), len(segs))
        otop, obot = _per_strip(op.get('ortho_top', 0.0), len(segs)), _per_strip(op.get('ortho_bottom', 0.0), len(segs))
        masks, profiles = [], []
        for a, b in segs:
            dist, s = _segment_distance_and_s(lx, ly, a, b)
            masks.append(dist <= half_w)
            profiles.append(_profile(op, s))
        for _ in range(int(tp['repeat'])):
            for k in range(len(segs)):
                q = np.exp(-beta * count) * masks[k]
                top_par = q * gtop[k] * profiles[k] * (1.0 + otop[k])
                top_perp = q * gtop[k] * profiles[k] * (1.0 - otop[k])
                bot_par = q * gbot[k] * (1.0 + obot[k])
                bot_perp = q * gbot[k] * (1.0 - obot[k])
                out['top_trace'] += top_par + top_perp
                out['bottom_trace'] += bot_par + bot_perp
                if o in ('A', 'B'):
                    out[f'along_{o}'] += 0.5 * (top_par + bot_par)
                    out[f'across_{o}'] += 0.5 * (top_perp + bot_perp)
                count = count + masks[k]
    out['hits'] = count
    return out


def treatment_monotonicity_violation(recipe: Dict, n_angles: int = 72, step_mm: float = 0.25,
                                     offset_mm: float = 0.0123456) -> Dict[str, float]:
    """Largest outward increase (relative to the centre value) of the hit count and of the top and
    bottom growth traces along dense rays from the centre; 0 means exactly non-increasing."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    radii = np.concatenate(([0.0], offset_mm + np.arange(0.0, 200.0 + step_mm / 2, step_mm)))
    U = np.cos(angles)[:, None] * radii[None, :]
    V = np.sin(angles)[:, None] * radii[None, :]
    field = decode_treatment_field(recipe, U, V)
    result = {}
    for key in ('hits', 'top_trace', 'bottom_trace'):
        f = field[key]
        scale = max(float(f[:, 0].max()), 1e-30)
        result[key] = float(max(0.0, np.diff(f, axis=1).max()) / scale)
    return result


def monotonicity_defect_fraction(recipe: Dict, key: str = 'hits', step_mm: float = 1.0,
                                 roi_half: Tuple[float, float] = (80.0, 120.0), n_samples: int = 200) -> float:
    """Area fraction of the ROI where the decoded field exceeds its minimum over the inner part of the ray
    from the centre, i.e. where an outward increase ('notch') exists. 0 for exactly monotone layouts.
    Nearest-grid sampling of the exact simulator-faithful field; boundary lines are measure zero."""
    from scipy.interpolate import RegularGridInterpolator
    u = np.arange(-SHEET_HALF_U_MM, SHEET_HALF_U_MM + step_mm / 2, step_mm)
    v = np.arange(-SHEET_HALF_V_MM, SHEET_HALF_V_MM + step_mm / 2, step_mm)
    U, V = np.meshgrid(u, v)
    f = decode_treatment_field(recipe, U, V)[key]
    interp = RegularGridInterpolator((v, u), f, method='nearest')
    running_min = np.full(f.shape, np.inf)
    for t in np.linspace(0.0, 1.0, n_samples + 1)[:-1]:                 # strictly inside the ray
        running_min = np.minimum(running_min, interp(np.stack([V * t, U * t], axis=-1)))
    bad = f > running_min + 1e-9 * max(float(np.abs(f).max()), 1e-30)
    roi = (np.abs(U) <= roi_half[0]) & (np.abs(V) <= roi_half[1])
    return float(bad[roi].mean())


def operation_footprint(op: Dict) -> Tuple[float, float, float, float]:
    """Axis-aligned painted rectangle (u_lo, u_hi, v_lo, v_hi) in mm, as the simulator checks it."""
    x = ((op['n_strips'] - 2) * op['lv_mm'] * np.tan(np.radians(op['alpha_deg'])) + op['width_mm']) / 2
    y = (op['lv_mm'] + op['width_mm']) / 2
    a = np.radians(op['rotation_deg'])
    c, s = np.cos(a), np.sin(a)
    cx = op['center_uv_mm'][0] + op['shift_uv_mm'][0]
    cy = op['center_uv_mm'][1] + op['shift_uv_mm'][1]
    pts = [(cx + c * u - s * v, cy + s * u + c * v) for u in (-x, x) for v in (-y, y)]
    return (min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts))


def conventional_zigzag_op(pitch_mm: float, width_mm: float, n_strips: int, lv_mm: float = 140.0,
                           rotation_deg: float = 0.0) -> Dict:
    """A conventional graded-pitch candidate: n_strips parallel-ish strokes at spacing `pitch_mm`."""
    alpha = float(np.degrees(np.arctan(pitch_mm / lv_mm)))
    return {'type': 'zigzag', 'lv_mm': lv_mm, 'alpha_deg': alpha, 'n_strips': n_strips, 'width_mm': width_mm,
            'center_uv_mm': [0.0, 0.0], 'shift_uv_mm': [0.0, 0.0], 'shift_frame': 'material',
            'rotation_deg': rotation_deg, 'gtop': G_TOP_DEFAULT, 'gbot': G_BOT_DEFAULT,
            'ortho_top': ORTHO_TOP_DEFAULT, 'ortho_bottom': ORTHO_BOT_DEFAULT, 'profile': {'mode': 'uniform'}}


def conventional_zigzag_ripple(pitch_mm: float = 6.0, width_mm: float = 14.0, n_strips: int = 12) -> float:
    """Largest outward hit increase (fraction of the maximum) of a conventional overlapping zigzag along
    the u axis: > 0 shows the hit-count ripple that rules this construction out."""
    recipe = {'hardening': {'beta': BETA_VOCE},
              'toolpaths': [{'id': 'crosshatch_A_conv', 'enabled': True, 'repeat': 1,
                             'operation': conventional_zigzag_op(pitch_mm, width_mm, n_strips)}]}
    u = np.arange(0.0, 80.0, 0.25)
    h = decode_hit_field(recipe, u, np.zeros_like(u))
    return float(max(0.0, np.diff(h).max()) / max(h.max(), 1e-30))


@dataclass
class PitchGradedCrossHatch:
    """Generator for nested, pitch-graded orthogonal cross-hatch terrace toolpaths."""
    total_dose: float = 4.5           # combined effective Voce dose at the centre (face-shared), < 4.95
                                      # (the asymptote 4.95 caps the sweeps: 4.5 -> 11 hits -> 6 sweeps -> 3 layers per orientation)
    dose_balance: float = 0.5         # share of the centre sweeps given to orientation A, in (0, 1)
    center_pitch_a: float = 22.0      # mm, innermost terrace spacing of A layers
    center_pitch_b: float = 22.0      # mm, innermost terrace spacing of B layers
    edge_pitch_factor: float = 2.0    # outermost / innermost terrace spacing (>= 1)
    decay_power: float = 2.5          # grading exponent of the spacing profile (>= 1)
    aspect_ratio: float = 1.25        # along/cross half-extent ratio of every capsule (>= 1)
    n_layers: int = 6                 # nested layers per orientation (capped by the sweep count)
    profile_mode: str = 'uniform'     # 'uniform' (monotone treatment guaranteed) or 'center_peak'
    center_peak_power: float = 2.0
    center_peak_end_ratio: float = 0.4
    layout: str = 'terrace'           # 'terrace' (default): the exactly monotone realisation of a pitch-graded hatch
                                      # (nested centred strokes = layer-cake of the stroke staircase, see module doc);
                                      # 'stroke': separated tiles with the same staircase levels, but round-cap notches
    edge_dose_fraction: float = 0.3   # stroke layout: repeat count at the rim relative to the centre, in (0, 1]

    def __post_init__(self):
        if self.layout not in ('stroke', 'terrace'):
            raise ValueError("layout must be 'stroke' or 'terrace'")
        if not 0.0 < self.edge_dose_fraction <= 1.0:
            raise ValueError("edge_dose_fraction must be in (0, 1]")
        if self.decay_power < 1.0:
            raise ValueError("Decay power must be >= 1")
        if self.aspect_ratio < 1.0 or self.edge_pitch_factor < 1.0:
            raise ValueError("aspect_ratio and edge_pitch_factor must be >= 1")
        if self.center_pitch_a <= 0.0 or self.center_pitch_b <= 0.0 or self.n_layers < 1:
            raise ValueError("invalid layer configuration")
        if not 0.0 < self.dose_balance < 1.0:
            raise ValueError("dose_balance must be in (0, 1)")
        if self.profile_mode not in ('uniform', 'center_peak'):
            raise ValueError("profile_mode must be 'uniform' or 'center_peak'")
        self.hits_target = invert_voce_dose(self.total_dose)
        total_sweeps = max(2, -(-self.hits_target // HITS_PER_SWEEP))
        self.sweeps_a = int(min(total_sweeps - 1, max(1, round(self.dose_balance * total_sweeps))))
        self.sweeps_b = total_sweeps - self.sweeps_a
        self.hits_a = HITS_PER_SWEEP * self.sweeps_a
        self.hits_b = HITS_PER_SWEEP * self.sweeps_b
        self.hits_centre = self.hits_a + self.hits_b

    # -- geometry ------------------------------------------------------------------------------
    def envelope(self) -> Tuple[float, float]:
        return A_E_MM, B_E_MAX_MM

    def _layers(self, orientation: str) -> List[Dict]:
        """Layers listed OUTERMOST first. Capsule half extents: `cross` (across the strips) and
        `along` (along the strips); every capsule is convex, centred and contains all inner ones."""
        sweeps = self.sweeps_a if orientation == 'A' else self.sweeps_b
        pitch0 = self.center_pitch_a if orientation == 'A' else self.center_pitch_b
        n = max(1, min(self.n_layers, sweeps))
        i = np.arange(n)
        spacing = pitch0 * (1.0 + (self.edge_pitch_factor - 1.0) * (i / max(n - 1, 1)) ** self.decay_power)
        radii = np.cumsum(spacing)
        radii = radii * min(1.0, A_E_MM / radii[-1])              # fit the sheet, keep nesting and grading
        spacing = np.diff(radii, prepend=0.0)                      # realised terrace spacing
        base, extra = divmod(sweeps, n)
        out = []
        for k in range(n - 1, -1, -1):                              # outermost first
            cross = float(radii[k])
            along_cap = B_E_MAX_MM if orientation == 'A' else A_E_MM
            along = float(min(along_cap, max(self.aspect_ratio * cross, cross + 0.5 * MIN_LV_MM)))
            width = 2.0 * cross
            lv = max(MIN_LV_MM, 2.0 * along - width)
            out.append({'layer': n - 1 - k, 'repeat': int(base + (1 if k < extra else 0)),
                        'cross': cross, 'along': (lv + width) / 2.0, 'width': float(width), 'lv': float(lv),
                        'spacing': float(spacing[k]), 'n_strips': 2})
        return out


    def _strokes(self, orientation: str) -> List[Dict]:
        """Spatially separated hatch strokes of one orientation, ordered centre-outward (one side).

        Strokes TILE the sheet across their stepping axis (width = pitch, no overlap, no gap), so the hit
        count is piecewise constant across strokes and no overlap ripple exists. The stroke pitch is
        p(x) = p_c (1 + (f-1)(x/R)^alpha) at inner-edge distance x from the centre (R = half extent across
        the strokes), i.e. it GROWS outward; the sweep repeats decay outward as
        r = round(r_c (1 - (1-edge_dose_fraction)(x_mid/R)^alpha)) >= 1. Chord extents follow the elliptical
        envelope but never exceed the straight part of the next inner stroke (e_k <= L_{k-1}/2), so
        outer strokes are nested inside inner ones along their length.
        """
        is_a = orientation == 'A'
        sweeps = self.sweeps_a if is_a else self.sweeps_b
        pitch0 = self.center_pitch_a if is_a else self.center_pitch_b
        b_e = min(B_E_MAX_MM, self.aspect_ratio * A_E_MM)
        half = A_E_MM if is_a else b_e                      # half extent across the strokes
        along_max = b_e if is_a else A_E_MM                  # centre chord half extent
        env = (lambda x: along_max * np.sqrt(max(0.0, 1.0 - (x / half) ** 2)))
        side, x = [], 0.5 * pitch0
        while half - x > 1e-9:
            p = pitch0 * (1.0 + (self.edge_pitch_factor - 1.0) * (x / half) ** self.decay_power)
            rem = half - x
            if rem < p:                                      # remainder narrower than a pitch: absorb it
                if side:
                    side[-1] = (side[-1][0], side[-1][1] + rem)
                else:
                    side.append((x, rem))
                break
            if rem < 1.5 * p:
                p = rem
            side.append((x, p))
            x += p
        strokes = [{'k': 0, 'x_in': 0.0, 'width': float(pitch0)}]
        strokes += [{'k': i + 1, 'x_in': float(xi), 'width': float(w)} for i, (xi, w) in enumerate(side)]
        prev = None
        for s_ in strokes:
            cap = np.inf if prev is None else prev['half_extent'] - 0.5 * prev['width']   # = L_{k-1}/2
            floor = 0.5 * s_['width'] + 0.5 * MIN_LV_MM
            s_['half_extent'] = float(max(min(env(s_['x_in']), cap), floor))
            s_['lv'] = float(2.0 * s_['half_extent'] - s_['width'])
            s_['pos'] = float(s_['x_in'] + 0.5 * s_['width'] if s_['k'] else 0.0)
            frac = min(1.0, s_['pos'] / half)
            s_['repeat'] = int(max(1, round(sweeps * (1.0 - (1.0 - self.edge_dose_fraction)
                                                       * frac ** self.decay_power))))
            prev = s_
        return strokes

    def stroke_table(self) -> Dict[str, List[Dict]]:
        return {'A': self._strokes('A'), 'B': self._strokes('B')}

    def layer_table(self) -> Dict[str, List[Dict]]:
        return {'A': self._layers('A'), 'B': self._layers('B')}

    # -- recipe --------------------------------------------------------------------------------
    def _profile_block(self) -> Dict:
        if self.profile_mode == 'uniform':
            return {'mode': 'uniform'}
        return {'mode': 'center_peak', 'top_end_ratio': self.center_peak_end_ratio,
                'top_profile_power': self.center_peak_power}

    def _operation(self, rotation: float, shift, lv: float, width: float) -> Dict:
        return {'type': 'zigzag', 'lv_mm': round(lv, 4),
                'alpha_deg': 1.0,                      # unused for n_strips = 2 (no inclined strips)
                'n_strips': 2, 'width_mm': round(width - (TILE_GAP_MM if self.layout == 'stroke' else 0.0), 4),
                'center_uv_mm': [0.0, 0.0], 'shift_uv_mm': [round(float(shift[0]), 4), round(float(shift[1]), 4)],
                'shift_frame': 'material', 'rotation_deg': rotation,
                'gtop': G_TOP_DEFAULT, 'gbot': G_BOT_DEFAULT,
                'ortho_top': ORTHO_TOP_DEFAULT, 'ortho_bottom': ORTHO_BOT_DEFAULT,
                'profile': self._profile_block()}

    def generate_recipe(self) -> Dict:
        """Simulator-compatible sequence recipe (no root-level metadata)."""
        toolpaths = []
        for orientation, rotation in (('A', 0.0), ('B', 90.0)):
            if self.layout == 'terrace':
                for k, lay in enumerate(self._layers(orientation)):
                    toolpaths.append({'id': f'crosshatch_{orientation}_L{k}', 'enabled': True,
                                      'repeat': lay['repeat'],
                                      'operation': self._operation(rotation, (0.0, 0.0), lay['lv'], lay['width'])})
                continue
            for st in self._strokes(orientation):
                for sign, tag in ((1.0, 'p'), (-1.0, 'n')) if st['k'] else ((1.0, ''),):
                    pos = sign * st['pos']
                    shift = (pos, 0.0) if orientation == 'A' else (0.0, pos)
                    toolpaths.append({'id': f"crosshatch_{orientation}_S{st['k']}{tag}", 'enabled': True,
                                      'repeat': st['repeat'],
                                      'operation': self._operation(rotation, shift, st['lv'], st['width'])})
        return {
            "schema_version": 1,
            "units": {"length": "mm", "angle": "deg", "growth": "engineering_strain"},
            "hardening": {"model": "voce_decay", "history_variable": "hit_count", "beta": BETA_VOCE,
                          "floor": 0, "scope": "face_shared", "count_rule": "plastic_hits"},
            "defaults": {"start_mode": "left_bottom_up"},
            "boundary_conditions": {"enabled": False},
            "toolpaths": toolpaths,
        }

    def parameters(self) -> Dict:
        return {"total_dose": self.total_dose, "dose_balance": self.dose_balance,
                "hits_target": self.hits_target, "hits_centre": self.hits_centre,
                "sweeps_a": self.sweeps_a, "sweeps_b": self.sweeps_b,
                "center_pitch_a": self.center_pitch_a, "center_pitch_b": self.center_pitch_b,
                "edge_pitch_factor": self.edge_pitch_factor, "decay_power": self.decay_power,
                "aspect_ratio": self.aspect_ratio, "n_layers": self.n_layers,
                "profile_mode": self.profile_mode, "layout": self.layout,
                "edge_dose_fraction": self.edge_dose_fraction}

    # -- surrogate coordinates of the emitted recipe -------------------------------------------
    def requested_raw(self) -> np.ndarray:
        """Surrogate coordinates as requested: [total_dose, balance, alpha, aspect, pitch_a, pitch_b]."""
        return np.array([self.total_dose, self.dose_balance, self.decay_power, self.aspect_ratio,
                         self.center_pitch_a, self.center_pitch_b], float)

    def realized_raw(self, recipe: Dict = None) -> np.ndarray:
        """Surrogate coordinates measured on the emitted recipe (integer sweeps, sheet fitting and
        the capsule aspect cap make them differ from the request). `decay_power` is the generator
        parameter; the pitches are the measured innermost terrace half-widths."""
        return raw_from_recipe(recipe or self.generate_recipe(), self.decay_power)

    # -- prescreen on the faithfully decoded treatment field -----------------------------------
    def compute_prescreen(self, grid_size: int = 101, recipe: Dict = None) -> Dict:
        """Heuristic Gauss-Codazzi ranking screen on the simulator-faithful treatment field.

        Mid-plane growth along/across the strips of each orientation (with face-shared hardening and
        the emitted profile) gives eps_vv = along_A + across_B and eps_uu = across_A + along_B; the
        small-strain, zero-shear target Gaussian curvature is K = -(1/2)(d2 eps_uu/dv2 + d2 eps_vv/du2).
        It is a heuristic for ranking only: post-solve curvature governs.
        """
        from scipy.ndimage import gaussian_filter
        recipe = recipe or self.generate_recipe()
        u = np.linspace(-SHEET_HALF_U_MM, SHEET_HALF_U_MM, grid_size)
        v = np.linspace(-SHEET_HALF_V_MM, SHEET_HALF_V_MM, int(round(grid_size * 1.5)))
        du, dv = u[1] - u[0], v[1] - v[0]
        U, V = np.meshgrid(u, v)
        f = decode_treatment_field(recipe, U, V)
        eps_vv = gaussian_filter(f['along_A'] + f['across_B'], sigma=(3.0 / dv, 3.0 / du))
        eps_uu = gaussian_filter(f['across_A'] + f['along_B'], sigma=(3.0 / dv, 3.0 / du))
        d2_uu_dv2 = np.gradient(np.gradient(eps_uu, dv, axis=0), dv, axis=0)
        d2_vv_du2 = np.gradient(np.gradient(eps_vv, du, axis=1), du, axis=1)
        K = -0.5 * (d2_uu_dv2 + d2_vv_du2)
        roi = (np.abs(U) <= 80.0) & (np.abs(V) <= 120.0)
        K_roi = K[roi]
        scale = np.max(np.abs(K_roi)) + 1e-30
        floor = 1e-3 * scale
        # Gaussian curvature is invariant under normal reversal: K > 0 is a dome/bowl (synclastic), K < 0 a saddle.
        pos = float(np.sum(np.maximum(0.0, K_roi)))
        neg = float(np.sum(np.maximum(0.0, -K_roi)))
        return {
            "synclastic_fraction": float(np.mean(K_roi > floor)),
            "saddle_fraction": float(np.mean(K_roi < -floor)),
            "prescreen_score": pos / (pos + neg + 1e-30),
            "target_K_pos_flux": pos * du * dv,
            "target_K_neg_flux": neg * du * dv,
        }


def raw_from_recipe(recipe: Dict, decay_power: float) -> np.ndarray:
    """[total_dose, balance, decay_power, aspect, pitch_a, pitch_b] measured on an emitted recipe.

    total_dose: Voce effective dose of the CENTRE hit count; balance: A share of the centre hits;
    pitch_o: width of the centre stroke of orientation o (stroke layout) or twice the innermost terrace
    half-width... i.e. the innermost spacing (terrace layout); aspect: stroke layout - centre chord half
    extent / sheet half width; terrace layout - along/cross extent of the outermost A capsule."""
    centre = {o: float(decode_hit_field(recipe, np.array([0.0]), np.array([0.0]), o)[0]) for o in 'AB'}
    total = centre['A'] + centre['B']
    ops = {o: [tp for tp in recipe['toolpaths'] if tp['id'].split('_')[1] == o] for o in 'AB'}
    stroke = all('_S' in tp['id'] for tp in recipe['toolpaths'])
    if stroke:
        pitch, extent = {}, None
        for o in 'AB':
            c = min(ops[o], key=lambda tp: abs(sum(tp['operation']['shift_uv_mm'])))
            pitch[o] = c['operation']['width_mm']
            if o == 'A':
                extent = 0.5 * (c['operation']['lv_mm'] + c['operation']['width_mm'])
        aspect = extent / A_E_MM
    else:
        pitch = {o: min(0.5 * tp['operation']['width_mm'] for tp in ops[o]) for o in 'AB'}
        outer = max(ops['A'], key=lambda tp: tp['operation']['width_mm'])['operation']
        aspect = (0.5 * (outer['lv_mm'] + outer['width_mm'])) / (0.5 * outer['width_mm'])
    return np.array([voce_effective_dose(int(round(total))), centre['A'] / total, decay_power, aspect,
                     pitch['A'], pitch['B']], float)


def generator_from_raw(raw, **kwargs) -> 'PitchGradedCrossHatch':
    """Generator for a surrogate proposal [total_dose, balance, decay_power, aspect, pitch_a, pitch_b].

    The proposal is a REQUEST: integer sweeps, sheet fitting and the capsule cap realise a nearby recipe.
    The surrogate must always be updated with ``generator.realized_raw()`` (measured on the emitted
    recipe), never with the request."""
    raw = np.asarray(raw, float)
    return PitchGradedCrossHatch(total_dose=float(raw[0]), dose_balance=float(raw[1]), decay_power=float(raw[2]),
                                 aspect_ratio=float(raw[3]), center_pitch_a=float(raw[4]),
                                 center_pitch_b=float(raw[5]), **kwargs)
