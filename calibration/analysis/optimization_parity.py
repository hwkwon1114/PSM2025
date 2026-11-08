#!/usr/bin/env python3
"""Four-tier result equivalence verifier for single-core solver optimizations.

Verifies that candidate optimizations preserve scientific and numerical results
relative to a locked baseline across three optimization invariance classes:

- Class A1 (Bit-Exact): Caching invariant quantities (aform_bar_inv, rest area,
  static triangle topology, CSC offset maps, CHOLMOD symbolic analysis reuse,
  preallocated gradient buffers). Requires 0.0 numerical difference and identical
  iteration/evaluation/Hessian counts.
- Class A2 (Algebraically Equivalent / IEEE-754 Reassociation): Fused top+bottom
  bilayer energy/gradient pass. Requires relative energy diff <= 1e-11,
  rigid-body-aligned 3D vertex diff <= 1e-9 mm, scan RMS diff <= 1e-9 mm, and
  iteration count within +/- 2%.
- Class B (Trajectory-Altering Algorithmic): Guarded Hessian reuse, adaptive
  L-BFGS-to-Newton handoff, or metricStrainJump initial continuation step sizing.
  Allows fewer iterations/evaluations/substeps, while requiring terminal
  acceptance ('gradient_target', 'positive_pivots', ||g|| <= 5e-14), relative
  energy diff <= 1e-10, rigid-body-aligned 3D vertex diff <= 1e-6 mm, and scan
  RMS diff <= 1e-6 mm.
"""
import argparse
import base64
from collections import namedtuple
import json
import math
import re
import struct
import sys
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


class EquivalenceThresholds(namedtuple('EquivalenceThresholds', [
    'opt_class',
    'rel_energy_tol',
    'max_aligned_vertex_mm',
    'rms_aligned_vertex_mm',
    'max_curvature_diff',
    'scan_rms_diff_mm',
    'require_exact_counts',
    'max_relative_attempt_drift',
    'gradient_tolerance',
])):
    def __new__(
        cls,
        opt_class,
        rel_energy_tol,
        max_aligned_vertex_mm,
        rms_aligned_vertex_mm,
        max_curvature_diff,
        scan_rms_diff_mm,
        require_exact_counts,
        max_relative_attempt_drift,
        gradient_tolerance=5e-14,
    ):
        return super(EquivalenceThresholds, cls).__new__(
            cls,
            opt_class,
            rel_energy_tol,
            max_aligned_vertex_mm,
            rms_aligned_vertex_mm,
            max_curvature_diff,
            scan_rms_diff_mm,
            require_exact_counts,
            max_relative_attempt_drift,
            gradient_tolerance,
        )


CLASS_THRESHOLDS: Dict[str, EquivalenceThresholds] = {
    'A1': EquivalenceThresholds(
        opt_class='A1',
        rel_energy_tol=0.0,
        max_aligned_vertex_mm=0.0,
        rms_aligned_vertex_mm=0.0,
        max_curvature_diff=0.0,
        scan_rms_diff_mm=0.0,
        require_exact_counts=True,
        max_relative_attempt_drift=0.0,
    ),
    'A2': EquivalenceThresholds(
        opt_class='A2',
        rel_energy_tol=1e-11,
        max_aligned_vertex_mm=1e-9,
        rms_aligned_vertex_mm=1e-10,
        max_curvature_diff=1e-7,
        scan_rms_diff_mm=1e-9,
        require_exact_counts=False,
        max_relative_attempt_drift=0.02,
    ),
    'B': EquivalenceThresholds(
        opt_class='B',
        rel_energy_tol=1e-10,
        max_aligned_vertex_mm=1e-6,
        rms_aligned_vertex_mm=2e-7,
        max_curvature_diff=1e-5,
        scan_rms_diff_mm=1e-6,
        require_exact_counts=False,
        max_relative_attempt_drift=None,
    ),
}


def _is_finite_number(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def validate_native_proof_certificate(
    proof: Dict[str, Any],
    max_gradient_tolerance: float = 5e-14,
) -> List[str]:
    """Check that a native_hybrid proof record satisfies all acceptance gates."""
    errors: List[str] = []
    if proof.get('accepted') is not True:
        errors.append(f"accepted is {proof.get('accepted')!r}, expected True")
    if proof.get('termination') != 'gradient_target':
        errors.append(f"termination is {proof.get('termination')!r}, expected 'gradient_target'")
    if proof.get('curvature') != 'positive_pivots':
        errors.append(f"curvature is {proof.get('curvature')!r}, expected 'positive_pivots'")

    grad_norm = proof.get('gradient_norm')
    grad_tol = proof.get('gradient_tolerance')
    energy = proof.get('energy')

    if not _is_finite_number(grad_norm) or float(grad_norm) < 0.0:
        errors.append(f"invalid gradient_norm: {grad_norm!r}")
    elif float(grad_norm) > max_gradient_tolerance:
        errors.append(
            f"gradient_norm {float(grad_norm):.3e} exceeds tolerance {max_gradient_tolerance:.3e}"
        )

    if not _is_finite_number(grad_tol) or float(grad_tol) <= 0.0 or float(grad_tol) > max_gradient_tolerance:
        errors.append(f"invalid gradient_tolerance: {grad_tol!r}")

    if not _is_finite_number(energy):
        errors.append(f"non-finite energy: {energy!r}")

    for key in ('verification_evaluations', 'verification_hessians'):
        val = proof.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            errors.append(f"{key} must be an integer >= 1, got {val!r}")

    for key in ('attempts', 'evaluations', 'hessians'):
        val = proof.get(key)
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            errors.append(f"{key} must be an integer >= 1, got {val!r}")

    return errors


def load_proof_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarize_proof_timings(proof_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lbfgs_s = sum(float(r.get('lbfgs_seconds', 0.0)) for r in proof_rows)
    newton_s = sum(float(r.get('newton_seconds', 0.0)) for r in proof_rows)
    wasted_s = sum(
        float(r.get('lbfgs_seconds', 0.0)) + float(r.get('newton_seconds', 0.0))
        for r in proof_rows
        if not r.get('accepted', False)
    )
    total_attempts = sum(int(r.get('attempts', 0)) for r in proof_rows)
    total_evals = sum(int(r.get('evaluations', 0)) for r in proof_rows)
    total_hessians = sum(int(r.get('hessians', 0)) for r in proof_rows)
    return {
        'substeps': len(proof_rows),
        'accepted_substeps': sum(1 for r in proof_rows if r.get('accepted') is True),
        'rejected_substeps': sum(1 for r in proof_rows if not r.get('accepted', False)),
        'total_attempts': total_attempts,
        'total_evaluations': total_evals,
        'total_hessians': total_hessians,
        'lbfgs_seconds': lbfgs_s,
        'newton_seconds': newton_s,
        'total_solver_seconds': lbfgs_s + newton_s,
        'wasted_cutback_seconds': wasted_s,
    }


def compare_manifests_and_proofs(
    base_manifest: Dict[str, Any],
    cand_manifest: Dict[str, Any],
    opt_class: str = 'A1',
    base_proof_rows: Optional[Sequence[Dict[str, Any]]] = None,
    cand_proof_rows: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Compare sequence_result.json manifests and native_hybrid proof records."""
    if opt_class not in CLASS_THRESHOLDS:
        raise ValueError(f"Unknown optimization class {opt_class!r}; expected one of {tuple(CLASS_THRESHOLDS)}")
    thresh = CLASS_THRESHOLDS[opt_class]
    errors: List[str] = []

    for label, m in (('baseline', base_manifest), ('candidate', cand_manifest)):
        if m.get('completed') is not True:
            errors.append(f"{label} manifest not completed")
        if m.get('backend') != 'native_hybrid':
            errors.append(f"{label} backend is {m.get('backend')!r}, expected 'native_hybrid'")
        solves = m.get('solves', [])
        if not solves:
            errors.append(f"{label} manifest has no solves")
            continue
        for idx, solve in enumerate(solves):
            if solve.get('return_code') != 0:
                errors.append(f"{label} solve[{idx}] return_code={solve.get('return_code')}")
            proof = solve.get('native_hybrid', {})
            cert_errors = validate_native_proof_certificate(proof, thresh.gradient_tolerance)
            errors.extend(f"{label} solve[{idx}]: {e}" for e in cert_errors)

    if errors:
        return {'passed': False, 'opt_class': opt_class, 'errors': errors}

    base_solve = base_manifest['solves'][-1]['native_hybrid']
    cand_solve = cand_manifest['solves'][-1]['native_hybrid']

    e_base = float(base_solve['energy'])
    e_cand = float(cand_solve['energy'])
    abs_e_diff = abs(e_cand - e_base)
    rel_e_diff = abs_e_diff / max(abs(e_base), 1e-30)

    g_base = float(base_solve['gradient_norm'])
    g_cand = float(cand_solve['gradient_norm'])
    abs_g_diff = abs(g_cand - g_base)

    if thresh.opt_class == 'A1':
        if abs_e_diff != 0.0:
            errors.append(f"Class A1 requires bit-exact energy (diff={abs_e_diff:.3e})")
        if abs_g_diff != 0.0:
            errors.append(f"Class A1 requires bit-exact gradient_norm (diff={abs_g_diff:.3e})")
    else:
        if rel_e_diff > thresh.rel_energy_tol:
            errors.append(
                f"relative energy diff {rel_e_diff:.3e} exceeds Class {opt_class} tolerance {thresh.rel_energy_tol:.3e}"
            )

    if thresh.require_exact_counts:
        for key in ('attempts', 'evaluations', 'hessians', 'warmup_attempts', 'newton_attempts'):
            if base_solve.get(key) != cand_solve.get(key):
                errors.append(
                    f"Class A1 requires identical {key}: baseline={base_solve.get(key)} vs candidate={cand_solve.get(key)}"
                )
        if base_proof_rows is not None and cand_proof_rows is not None:
            if len(base_proof_rows) != len(cand_proof_rows):
                errors.append(
                    f"Class A1 requires identical substep count: baseline={len(base_proof_rows)} vs candidate={len(cand_proof_rows)}"
                )
    elif thresh.max_relative_attempt_drift is not None:
        b_att = int(base_solve['attempts'])
        c_att = int(cand_solve['attempts'])
        rel_att = abs(c_att - b_att) / max(1, b_att)
        if rel_att > thresh.max_relative_attempt_drift:
            errors.append(
                f"attempt count drift {rel_att:.3%} exceeds Class {opt_class} limit {thresh.max_relative_attempt_drift:.3%}"
            )

    timing_comparison = None
    if base_proof_rows is not None and cand_proof_rows is not None:
        base_t = summarize_proof_timings(base_proof_rows)
        cand_t = summarize_proof_timings(cand_proof_rows)
        speedup = (
            base_t['total_solver_seconds'] / max(1e-12, cand_t['total_solver_seconds'])
            if cand_t['total_solver_seconds'] > 0
            else None
        )
        timing_comparison = {
            'baseline': base_t,
            'candidate': cand_t,
            'solver_speedup_x': speedup,
        }

    return {
        'passed': len(errors) == 0,
        'opt_class': opt_class,
        'energy_baseline': e_base,
        'energy_candidate': e_cand,
        'abs_energy_diff': abs_e_diff,
        'rel_energy_diff': rel_e_diff,
        'gradient_norm_baseline': g_base,
        'gradient_norm_candidate': g_cand,
        'abs_gradient_norm_diff': abs_g_diff,
        'timing': timing_comparison,
        'errors': errors,
    }


_VTK_STRUCT_TYPES = {
    'Float64': ('d', 8),
    'Float32': ('f', 4),
    'Int64': ('q', 8),
    'UInt64': ('Q', 8),
    'Int32': ('i', 4),
    'UInt32': ('I', 4),
}


def _decode_vtp_data_array(
    da: ET.Element,
    appended_b64: Optional[str],
    header_type: str = 'UInt32',
    compressed: bool = True,
) -> List[Any]:
    dtype = da.attrib['type']
    fmt_char, elem_bytes = _VTK_STRUCT_TYPES[dtype]
    fmt = da.attrib.get('format', 'ascii')
    if fmt == 'ascii':
        text = da.text or ''
        if fmt_char in ('d', 'f'):
            return [float(x) for x in text.split()]
        return [int(x) for x in text.split()]

    if fmt != 'appended' or appended_b64 is None:
        raise ValueError(f"Unsupported VTP DataArray format={fmt!r}")

    off = int(da.attrib['offset'])
    hdr_fmt, hdr_size = _VTK_STRUCT_TYPES.get(header_type, ('I', 4))
    if compressed:
        first3_bytes = 3 * hdr_size
        first3_b64 = ((first3_bytes + 2) // 3) * 4
        nb, _bs, _lbs = struct.unpack(
            '<' + hdr_fmt * 3,
            base64.b64decode(appended_b64[off : off + first3_b64])[:first3_bytes],
        )
        total_hdr_bytes = (3 + nb) * hdr_size
        total_hdr_b64 = ((total_hdr_bytes + 2) // 3) * 4
        hdr = struct.unpack(
            '<' + hdr_fmt * (3 + nb),
            base64.b64decode(appended_b64[off : off + total_hdr_b64])[:total_hdr_bytes],
        )
        csizes = hdr[3:]
        payload_bytes = sum(csizes)
        payload_b64 = ((payload_bytes + 2) // 3) * 4
        comp = base64.b64decode(
            appended_b64[off + total_hdr_b64 : off + total_hdr_b64 + payload_b64]
        )
        chunks: List[bytes] = []
        pos = 0
        for cs in csizes:
            if cs > 0:
                chunks.append(zlib.decompress(comp[pos : pos + cs]))
                pos += cs
        raw = b''.join(chunks)
    else:
        hdr_b64 = ((hdr_size + 2) // 3) * 4
        (nbytes,) = struct.unpack(
            '<' + hdr_fmt,
            base64.b64decode(appended_b64[off : off + hdr_b64])[:hdr_size],
        )
        payload_b64 = ((nbytes + 2) // 3) * 4
        raw = base64.b64decode(appended_b64[off + hdr_b64 : off + hdr_b64 + payload_b64])[:nbytes]

    count = len(raw) // elem_bytes
    return list(struct.unpack('<' + fmt_char * count, raw))


def parse_ascii_vtp(path: Path) -> Dict[str, Any]:
    """Parse vertices, triangle connectivity, and scalar fields from an ASCII or appended VTP file."""
    root = ET.parse(path).getroot()
    piece = root.find('.//Piece')
    if piece is None:
        raise ValueError(f"Missing <Piece> element in VTP file {path}")

    header_type = root.attrib.get('header_type', 'UInt32')
    compressed = 'compressor' in root.attrib
    appended_el = root.find('AppendedData')
    appended_b64: Optional[str] = None
    if appended_el is not None and appended_el.text:
        txt = appended_el.text.strip()
        if txt.startswith('_'):
            appended_b64 = txt[1:]

    n_points = int(piece.attrib['NumberOfPoints'])
    n_polys = int(piece.attrib['NumberOfPolys'])

    points_da = piece.find('.//Points/DataArray')
    if points_da is None:
        raise ValueError(f"Missing <Points/DataArray> in {path}")
    raw_coords = _decode_vtp_data_array(points_da, appended_b64, header_type, compressed)
    if len(raw_coords) != 3 * n_points:
        raise ValueError(f"Expected {3 * n_points} point coordinates, got {len(raw_coords)}")

    # If double-precision U_from_cycle0 and material_u/material_v are stored in PointData,
    # also capture high-precision Float64 displacement vectors for exact ULP checks.
    u_da = piece.find(".//PointData/DataArray[@Name='U_from_cycle0']")
    displacements_f64: Optional[List[Tuple[float, float, float]]] = None
    if u_da is not None:
        raw_u = _decode_vtp_data_array(u_da, appended_b64, header_type, compressed)
        if len(raw_u) == 3 * n_points:
            displacements_f64 = [
                (float(raw_u[3 * i]), float(raw_u[3 * i + 1]), float(raw_u[3 * i + 2]))
                for i in range(n_points)
            ]

    vertices = [
        (float(raw_coords[3 * i]), float(raw_coords[3 * i + 1]), float(raw_coords[3 * i + 2]))
        for i in range(n_points)
    ]

    conn_da = piece.find(".//Polys/DataArray[@Name='connectivity']")
    off_da = piece.find(".//Polys/DataArray[@Name='offsets']")
    if conn_da is None or off_da is None:
        raise ValueError(f"Missing polygon connectivity/offsets in {path}")
    conn = [int(x) for x in _decode_vtp_data_array(conn_da, appended_b64, header_type, compressed)]
    offsets = [int(x) for x in _decode_vtp_data_array(off_da, appended_b64, header_type, compressed)]
    if len(offsets) != n_polys or len(conn) != 3 * n_polys:
        raise ValueError(f"Non-triangular or malformed Polys in {path}")

    cell_fields: Dict[str, List[float]] = {}
    cell_data = piece.find('CellData')
    if cell_data is not None:
        for da in cell_data.findall('DataArray'):
            name = da.attrib.get('Name')
            if name in ('mean', 'gauss', 'abar_top_11', 'abar_top_12', 'abar_top_22', 'abar_bot_11', 'abar_bot_12', 'abar_bot_22'):
                vals = [float(x) for x in _decode_vtp_data_array(da, appended_b64, header_type, compressed)]
                if len(vals) == n_polys:
                    cell_fields[name] = vals

    return {
        'vertices': vertices,
        'displacements_f64': displacements_f64,
        'connectivity': conn,
        'offsets': offsets,
        'cell_fields': cell_fields,
    }


def _mat3_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    return [
        [sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def _mat3_transpose(A: List[List[float]]) -> List[List[float]]:
    return [[A[j][i] for j in range(3)] for i in range(3)]


def _mat3_det(A: List[List[float]]) -> float:
    return (
        A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
        - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
        + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
    )


def _jacobi_eigen_sym3(S: List[List[float]]) -> Tuple[List[float], List[List[float]]]:
    """Exact 3x3 symmetric eigendecomposition via Jacobi rotations (pure Python)."""
    A = [row[:] for row in S]
    V = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for _ in range(50):
        # Find largest off-diagonal entry
        p, q = 0, 1
        max_val = abs(A[0][1])
        for i, j in ((0, 2), (1, 2)):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j
        if max_val < 1e-16:
            break
        diff = A[q][q] - A[p][p]
        if abs(A[p][q]) < 1e-30 * abs(diff):
            t = A[p][q] / diff
        else:
            phi = diff / (2.0 * A[p][q])
            t = 1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0))
            if phi < 0.0:
                t = -t
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c
        tau = s / (1.0 + c)
        h = t * A[p][q]
        A[p][p] -= h
        A[q][q] += h
        A[p][q] = A[q][p] = 0.0
        for r in range(3):
            if r != p and r != q:
                arp, arq = A[r][p], A[r][q]
                A[r][p] = A[p][r] = arp - s * (arq + tau * arp)
                A[r][q] = A[q][r] = arq + s * (arp - tau * arq)
        for r in range(3):
            vrp, vrq = V[r][p], V[r][q]
            V[r][p] = vrp - s * (vrq + tau * vrp)
            V[r][q] = vrq + s * (vrp - tau * vrq)
    evals = [A[0][0], A[1][1], A[2][2]]
    # Sort descending
    order = sorted(range(3), key=lambda k: evals[k], reverse=True)
    sorted_evals = [evals[k] for k in order]
    sorted_V = [[V[row][k] for k in order] for row in range(3)]
    if _mat3_det(sorted_V) < 0.0:
        for row in range(3):
            sorted_V[row][2] = -sorted_V[row][2]
    return sorted_evals, sorted_V


def kabsch_align_3d(
    ref_pts: Sequence[Tuple[float, float, float]],
    cand_pts: Sequence[Tuple[float, float, float]],
    scale_to_mm: float = 1000.0,
) -> Dict[str, float]:
    """Compute raw and 6-DOF Kabsch rigid-body-aligned vertex differences in mm."""
    n = len(ref_pts)
    if n != len(cand_pts) or n == 0:
        raise ValueError("Point sets must have identical non-zero length")

    raw_sq_sum = 0.0
    raw_max = 0.0
    for (rx, ry, rz), (cx, cy, cz) in zip(ref_pts, cand_pts):
        dx = (cx - rx) * scale_to_mm
        dy = (cy - ry) * scale_to_mm
        dz = (cz - rz) * scale_to_mm
        d2 = dx * dx + dy * dy + dz * dz
        raw_sq_sum += d2
        d = math.sqrt(d2)
        if d > raw_max:
            raw_max = d
    raw_rms = math.sqrt(raw_sq_sum / n)

    if raw_max == 0.0:
        return {
            'raw_max_vertex_diff_mm': 0.0,
            'raw_rms_vertex_diff_mm': 0.0,
            'aligned_max_vertex_diff_mm': 0.0,
            'aligned_rms_vertex_diff_mm': 0.0,
        }

    # Centroids in mm
    rcx = sum(p[0] for p in ref_pts) * scale_to_mm / n
    rcy = sum(p[1] for p in ref_pts) * scale_to_mm / n
    rcz = sum(p[2] for p in ref_pts) * scale_to_mm / n

    ccx = sum(p[0] for p in cand_pts) * scale_to_mm / n
    ccy = sum(p[1] for p in cand_pts) * scale_to_mm / n
    ccz = sum(p[2] for p in cand_pts) * scale_to_mm / n

    P = [(p[0] * scale_to_mm - ccx, p[1] * scale_to_mm - ccy, p[2] * scale_to_mm - ccz) for p in cand_pts]
    Q = [(q[0] * scale_to_mm - rcx, q[1] * scale_to_mm - rcy, q[2] * scale_to_mm - rcz) for q in ref_pts]

    # Covariance H = P^T Q
    H = [[0.0, 0.0, 0.0] for _ in range(3)]
    for p, q in zip(P, Q):
        for i in range(3):
            for j in range(3):
                H[i][j] += p[i] * q[j]

    HT = _mat3_transpose(H)
    HTH = _mat3_mul(HT, H)
    evals, V = _jacobi_eigen_sym3(HTH)

    # Compute U = H V Sigma^{-1} via orthonormal columns
    HV = _mat3_mul(H, V)
    U = [[0.0, 0.0, 0.0] for _ in range(3)]
    for col in range(2):
        norm = math.sqrt(sum(HV[row][col] ** 2 for row in range(3)))
        if norm > 1e-15:
            for row in range(3):
                U[row][col] = HV[row][col] / norm
        else:
            U[col][col] = 1.0
    # Orthogonalize col 1 against col 0
    dot01 = sum(U[row][0] * U[row][1] for row in range(3))
    for row in range(3):
        U[row][1] -= dot01 * U[row][0]
    norm1 = math.sqrt(sum(U[row][1] ** 2 for row in range(3)))
    if norm1 > 1e-15:
        for row in range(3):
            U[row][1] /= norm1
    # Col 2 = col 0 x col 1 (ensures det(U) = +1)
    U[0][2] = U[1][0] * U[2][1] - U[2][0] * U[1][1]
    U[1][2] = U[2][0] * U[0][1] - U[0][0] * U[2][1]
    U[2][2] = U[0][0] * U[1][1] - U[1][0] * U[0][1]

    # Optimal proper rotation R = V U^T (maps column vectors p -> q)
    R = _mat3_mul(V, _mat3_transpose(U))
    if _mat3_det(R) < 0.0:
        V_flipped = [row[:] for row in V]
        for row in range(3):
            V_flipped[row][2] = -V_flipped[row][2]
        R = _mat3_mul(V_flipped, _mat3_transpose(U))

    aligned_sq_sum = 0.0
    aligned_max = 0.0
    for p, q in zip(P, Q):
        rx = R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2]
        ry = R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2]
        rz = R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2]
        dx = rx - q[0]
        dy = ry - q[1]
        dz = rz - q[2]
        d2 = dx * dx + dy * dy + dz * dz
        aligned_sq_sum += d2
        d = math.sqrt(d2)
        if d > aligned_max:
            aligned_max = d

    return {
        'raw_max_vertex_diff_mm': raw_max,
        'raw_rms_vertex_diff_mm': raw_rms,
        'aligned_max_vertex_diff_mm': aligned_max,
        'aligned_rms_vertex_diff_mm': math.sqrt(aligned_sq_sum / n),
    }


def compare_vtp_meshes(
    base_vtp: Path,
    cand_vtp: Path,
    opt_class: str = 'A1',
    scale_to_mm: float = 1000.0,
) -> Dict[str, Any]:
    """Compare two VTP meshes across connectivity, aligned 3D vertices, and curvatures."""
    if opt_class not in CLASS_THRESHOLDS:
        raise ValueError(f"Unknown optimization class {opt_class!r}")
    thresh = CLASS_THRESHOLDS[opt_class]
    errors: List[str] = []

    m_base = parse_ascii_vtp(base_vtp)
    m_cand = parse_ascii_vtp(cand_vtp)

    if len(m_base['vertices']) != len(m_cand['vertices']):
        errors.append(
            f"Vertex count mismatch: {len(m_base['vertices'])} vs {len(m_cand['vertices'])}"
        )
        return {'passed': False, 'opt_class': opt_class, 'errors': errors}

    if m_base['connectivity'] != m_cand['connectivity'] or m_base['offsets'] != m_cand['offsets']:
        errors.append("Triangle connectivity or offsets do not match identically")
        return {'passed': False, 'opt_class': opt_class, 'errors': errors}

    v_metrics = kabsch_align_3d(m_base['vertices'], m_cand['vertices'], scale_to_mm=scale_to_mm)

    if thresh.opt_class == 'A1':
        if v_metrics['raw_max_vertex_diff_mm'] != 0.0:
            errors.append(
                f"Class A1 requires bit-exact vertex coordinates, got raw_max_diff={v_metrics['raw_max_vertex_diff_mm']:.3e} mm"
            )
    else:
        if v_metrics['aligned_max_vertex_diff_mm'] > thresh.max_aligned_vertex_mm:
            errors.append(
                f"Aligned max vertex diff {v_metrics['aligned_max_vertex_diff_mm']:.3e} mm exceeds "
                f"Class {opt_class} limit {thresh.max_aligned_vertex_mm:.3e} mm"
            )
        if v_metrics['aligned_rms_vertex_diff_mm'] > thresh.rms_aligned_vertex_mm:
            errors.append(
                f"Aligned RMS vertex diff {v_metrics['aligned_rms_vertex_diff_mm']:.3e} mm exceeds "
                f"Class {opt_class} limit {thresh.rms_aligned_vertex_mm:.3e} mm"
            )

    curvature_diffs: Dict[str, float] = {}
    for field in ('mean', 'gauss'):
        if field in m_base['cell_fields'] and field in m_cand['cell_fields']:
            b_vals = m_base['cell_fields'][field]
            c_vals = m_cand['cell_fields'][field]
            max_diff = max(abs(a - b) for a, b in zip(b_vals, c_vals))
            curvature_diffs[f'max_{field}_diff'] = max_diff
            if max_diff > thresh.max_curvature_diff:
                errors.append(
                    f"Cell field '{field}' max diff {max_diff:.3e} exceeds Class {opt_class} limit {thresh.max_curvature_diff:.3e}"
                )

    return {
        'passed': len(errors) == 0,
        'opt_class': opt_class,
        'vertex_count': len(m_base['vertices']),
        'face_count': len(m_base['offsets']),
        **v_metrics,
        **curvature_diffs,
        'errors': errors,
    }


def compare_evaluation_records(
    base_eval: Dict[str, Any],
    cand_eval: Dict[str, Any],
    opt_class: str = 'A1',
) -> Dict[str, Any]:
    """Compare two single-candidate evaluation records (e.g. from evaluations.jsonl)."""
    if opt_class not in CLASS_THRESHOLDS:
        raise ValueError(f"Unknown optimization class {opt_class!r}")
    thresh = CLASS_THRESHOLDS[opt_class]
    errors: List[str] = []

    base_valid = bool(base_eval.get('valid'))
    cand_valid = bool(cand_eval.get('valid'))

    if base_valid != cand_valid:
        errors.append(f"Validity mismatch: baseline valid={base_valid} vs candidate valid={cand_valid}")
        return {'passed': False, 'opt_class': opt_class, 'errors': errors}

    if not base_valid:
        # Both invalid (e.g. known solver failure candidate): ensure failure class is preserved
        b_cls = base_eval.get('failure_class')
        c_cls = cand_eval.get('failure_class')
        if b_cls != c_cls:
            errors.append(f"Failure class mismatch on invalid candidate: {b_cls!r} vs {c_cls!r}")
        return {
            'passed': len(errors) == 0,
            'opt_class': opt_class,
            'valid': False,
            'failure_class': c_cls,
            'errors': errors,
        }

    b_obj = float(base_eval['objective_mm'])
    c_obj = float(cand_eval['objective_mm'])
    obj_diff = abs(c_obj - b_obj)

    if obj_diff > thresh.scan_rms_diff_mm:
        errors.append(
            f"Objective RMS diff {obj_diff:.3e} mm exceeds Class {opt_class} tolerance {thresh.scan_rms_diff_mm:.3e} mm"
        )

    manifest_report = None
    if base_eval.get('cases') and cand_eval.get('cases'):
        b_case = base_eval['cases'][0]
        c_case = cand_eval['cases'][0]
        if 'manifest' in b_case and 'manifest' in c_case:
            manifest_report = compare_manifests_and_proofs(
                b_case['manifest'], c_case['manifest'], opt_class=opt_class
            )
            if not manifest_report['passed']:
                errors.extend(manifest_report['errors'])

    b_sec = float(base_eval.get('seconds', 0.0))
    c_sec = float(cand_eval.get('seconds', 0.0))
    wall_speedup = (b_sec / c_sec) if c_sec > 0 else None

    return {
        'passed': len(errors) == 0,
        'opt_class': opt_class,
        'valid': True,
        'objective_baseline_mm': b_obj,
        'objective_candidate_mm': c_obj,
        'abs_objective_diff_mm': obj_diff,
        'wall_seconds_baseline': b_sec,
        'wall_seconds_candidate': c_sec,
        'wall_speedup_x': wall_speedup,
        'manifest_report': manifest_report,
        'errors': errors,
    }


def verify_multi_point_suite(
    baseline_records: Sequence[Dict[str, Any]],
    candidate_records: Sequence[Dict[str, Any]],
    opt_class: str = 'B',
) -> Dict[str, Any]:
    """Tier 4 multi-specimen & edge-case suite verification (including rank-order gate)."""
    if len(baseline_records) != len(candidate_records) or not baseline_records:
        raise ValueError("Baseline and candidate suites must have identical non-zero length")

    point_reports = [
        compare_evaluation_records(b, c, opt_class=opt_class)
        for b, c in zip(baseline_records, candidate_records)
    ]
    errors: List[str] = []
    for idx, rep in enumerate(point_reports):
        if not rep['passed']:
            errors.extend(f"point[{idx}]: {e}" for e in rep['errors'])

    # Check rank-order invariance among valid points whose objectives are separated by > 2e-6 mm
    valid_indices = [
        i for i in range(len(baseline_records))
        if baseline_records[i].get('valid') and candidate_records[i].get('valid')
    ]
    rank_swaps = 0
    for idx_a in range(len(valid_indices)):
        for idx_b in range(idx_a + 1, len(valid_indices)):
            i = valid_indices[idx_a]
            j = valid_indices[idx_b]
            b_diff = float(baseline_records[i]['objective_mm']) - float(baseline_records[j]['objective_mm'])
            c_diff = float(candidate_records[i]['objective_mm']) - float(candidate_records[j]['objective_mm'])
            if abs(b_diff) > 2e-6 and (b_diff * c_diff < 0.0):
                rank_swaps += 1
                errors.append(
                    f"Rank inversion between point[{i}] and point[{j}]: "
                    f"baseline dJ={b_diff:.6f} mm vs candidate dJ={c_diff:.6f} mm"
                )

    return {
        'passed': len(errors) == 0,
        'opt_class': opt_class,
        'points_tested': len(baseline_records),
        'rank_swaps': rank_swaps,
        'point_reports': point_reports,
        'errors': errors,
    }


def verify_case_directories(
    base_dir: Path,
    cand_dir: Path,
    opt_class: str = 'A1',
) -> Dict[str, Any]:
    """Compare two solver case_00 directories (sequence_result.json + proof.jsonl + final VTP)."""
    base_manifest = json.loads((base_dir / 'sequence_result.json').read_text())
    cand_manifest = json.loads((cand_dir / 'sequence_result.json').read_text())

    base_proof_path = base_dir / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl'
    cand_proof_path = cand_dir / 'bilayer_zigzag_sequence_native_hybrid_proof.jsonl'
    base_proofs = load_proof_jsonl(base_proof_path) if base_proof_path.is_file() else None
    cand_proofs = load_proof_jsonl(cand_proof_path) if cand_proof_path.is_file() else None

    m_rep = compare_manifests_and_proofs(
        base_manifest, cand_manifest, opt_class=opt_class,
        base_proof_rows=base_proofs, cand_proof_rows=cand_proofs,
    )

    base_vtp = base_dir / base_manifest['final_mesh']
    cand_vtp = cand_dir / cand_manifest['final_mesh']
    vtp_rep = compare_vtp_meshes(base_vtp, cand_vtp, opt_class=opt_class)

    errors = list(m_rep['errors']) + list(vtp_rep['errors'])
    return {
        'passed': len(errors) == 0,
        'opt_class': opt_class,
        'manifest_and_proof': m_rep,
        'mesh_parity': vtp_rep,
        'errors': errors,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class', dest='opt_class', choices=('A1', 'A2', 'B'), default='A1')
    parser.add_argument('--baseline', type=Path, required=True, help='Baseline case directory or evaluations.jsonl')
    parser.add_argument('--candidate', type=Path, required=True, help='Candidate case directory or evaluations.jsonl')
    parser.add_argument('--output', type=Path, default=None, help='Optional JSON report output path')
    args = parser.parse_args(argv)

    if args.baseline.is_dir() and args.candidate.is_dir():
        report = verify_case_directories(args.baseline, args.candidate, opt_class=args.opt_class)
    else:
        b_rec = json.loads(args.baseline.read_text().splitlines()[-1])
        c_rec = json.loads(args.candidate.read_text().splitlines()[-1])
        report = compare_evaluation_records(b_rec, c_rec, opt_class=args.opt_class)

    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
