#!/usr/bin/env python3
"""Export displacement fields and render shape / toolpath pictures for solver-characteristics cases.

Everything written by this script comes from solver output that already exists on disk
(``*_final.vtp``, ``*_mapping.vtp``, ``sequence.json``, ``manifest.json``, ``result.json``).
No simulation is launched from here.

Two different things are produced and never mixed up:

* *current shape*  -- the deformed vertex coordinates stored in the VTP (``x``, ``y``, ``z``).
* *displacement*   -- ``U = x_current - x_reference`` with the reference being the
  ``cycle_000_initial`` flat plate, i.e. the solver's ``U_from_cycle0`` point array.
  Because the sequence runs with a numerical rigid-body gauge (no physical clamps),
  the raw displacement contains rigid-body motion.  A second, explicitly labelled
  rigid-aligned displacement (proper Kabsch alignment of the current shape onto the
  reference) is exported next to it.

Toolpath pictures use the *exact* zigzag geometry of the executed recipe, ported
one-to-one from ``src/libshell/ZigZagGrowth.hpp`` (``buildZigZagStrips`` plus the
placement transform of ``collectMaterialCoordinateHits``), and each reconstruction is
validated against the solver's own per-face mapping arrays before it is plotted.  The
discrete per-face contact map from the mapping VTP is drawn underneath it, so the
figure shows measured solver contact and reconstructed centreline together.

All exported lengths are in metres (SI).  Figure axes are annotated in millimetres,
which is the unit convention used by the rest of the repository's figures.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import tri as mtri
from matplotlib.collections import PolyCollection

from analyze_sequence_ablation_v2 import (
    areas,
    cell_array,
    final_vtp,
    load_vtp,
    point_array,
    proper_kabsch,
    quadratic_fit,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / "run" / "equilibrium_continuation_study"
OUT_ROOT = ROOT / "run" / "solver_characteristics" / "figures"

def rel(path: Path) -> str:
    """Repository-relative path when possible, absolute otherwise."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


CYCLE_RE = re.compile(r"_cycle_(\d{3})_(.+)_r(\d{3})_(final|mapping)\.vtp$")
INITIAL_RE = re.compile(r"_cycle_(\d{3})_initial\.vtp$")

FIELD_COLUMNS = [
    "point_index",
    "material_u_m",
    "material_v_m",
    "reference_x_m",
    "reference_y_m",
    "reference_z_m",
    "current_x_m",
    "current_y_m",
    "current_z_m",
    "U_raw_x_m",
    "U_raw_y_m",
    "U_raw_z_m",
    "U_raw_magnitude_m",
    "U_rigid_aligned_x_m",
    "U_rigid_aligned_y_m",
    "U_rigid_aligned_z_m",
    "U_rigid_aligned_magnitude_m",
]

COLUMN_DOC = {
    "point_index": "zero-based VTP point index (identical ordering in every cycle of one case)",
    "material_u_m / material_v_m": "persistent material coordinates, VTP point arrays material_u/material_v, metres",
    "reference_*_m": "vertex coordinates of the cycle_000_initial flat plate, metres",
    "current_*_m": "vertex coordinates of this cycle's VTP, i.e. the CURRENT SHAPE, metres",
    "U_raw_*_m": "solver point array U_from_cycle0 = current - reference, metres, INCLUDES rigid-body motion",
    "U_raw_magnitude_m": "Euclidean norm of U_raw, metres (equals solver array Umag_from_cycle0)",
    "U_rigid_aligned_*_m": "displacement after proper Kabsch alignment of the current shape onto the reference, metres, rigid-body motion REMOVED",
    "U_rigid_aligned_magnitude_m": "Euclidean norm of U_rigid_aligned, metres",
}


# ---------------------------------------------------------------------------
# exact port of the C++ zigzag geometry (src/libshell/ZigZagGrowth.hpp)
# ---------------------------------------------------------------------------
def normalize_angle_pi(angle: float) -> float:
    """Port of zigzag::normalize_angle_pi: orientation in [0, pi]."""
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    if angle < 0.0:
        angle += math.pi
    return angle


def rot2d(angle_rad: float) -> np.ndarray:
    cos, sin = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[cos, -sin], [sin, cos]])


def build_zigzag_strips(lv_mm: float, alpha_deg: float, n_strips: int) -> list:
    """Port of zigzag::buildZigZagStrips: local-frame strip segments, metres."""
    if n_strips < 2:
        raise ValueError("n_strips must be >= 2")
    n_incl = n_strips - 2
    lv = lv_mm * 1e-3
    alpha = math.radians(alpha_deg)
    pitch = lv * math.tan(alpha)
    y_bot, y_top = -0.5 * lv, 0.5 * lv
    span = n_incl * pitch if n_incl > 0 else 0.0
    x_left, x_right = -0.5 * span, 0.5 * span

    pts = [np.array([x_left, y_bot]), np.array([x_left, y_top])]
    for j in range(1, n_incl + 1):
        pts.append(np.array([x_left + j * pitch, y_bot if j % 2 == 1 else y_top]))
    y_other = y_bot if abs(pts[-1][1] - y_top) < 1e-15 else y_top
    pts.append(np.array([x_right, y_other]))

    segments = [(0, pts[0], pts[1])]
    segments.extend((k, pts[k], pts[k + 1]) for k in range(1, n_incl + 1))
    segments.append((n_strips - 1, pts[-2], pts[-1]))

    strips = []
    for index, start, end in segments:
        direction = end - start
        strips.append({
            "strip_index": index,
            "a_local_m": start,
            "b_local_m": end,
            "angle_local_rad": normalize_angle_pi(math.atan2(direction[1], direction[0])),
        })
    return strips


def place_strips(operation: dict, material_bbox: tuple) -> dict:
    """Port of the placement used by zigzag::collectMaterialCoordinateHits.

    p_uv = center + R(rotation) @ p_local + translation
    """
    n_strips = int(operation["n_strips"])
    strips = build_zigzag_strips(float(operation["lv_mm"]), float(operation["alpha_deg"]), n_strips)
    half_width = 0.5 * float(operation["width_mm"]) * 1e-3

    umin, umax, vmin, vmax = material_bbox
    material_center = np.array([0.5 * (umin + umax), 0.5 * (vmin + vmax)])
    if "center_uv_mm" in operation:
        center = np.asarray(operation["center_uv_mm"], dtype=float) * 1e-3
        center_source = "operation.center_uv_mm"
    else:
        center = material_center
        center_source = "material bounding-box centre"

    rotation_rad = math.radians(float(operation.get("rotation_deg", 0.0)))
    rotation = rot2d(rotation_rad)
    shift = np.asarray(operation.get("shift_uv_mm", (0.0, 0.0)), dtype=float) * 1e-3
    if str(operation.get("shift_frame", "material")).lower() == "pattern":
        translation = rotation @ shift
    else:
        translation = shift

    active = operation.get("active_strips")
    active_set = set(int(index) for index in active) if active else None

    placed = []
    for strip in strips:
        if active_set is not None and strip["strip_index"] not in active_set:
            continue
        placed.append({
            "strip_index": strip["strip_index"],
            "a_uv_m": center + rotation @ strip["a_local_m"] + translation,
            "b_uv_m": center + rotation @ strip["b_local_m"] + translation,
            "angle_material_rad": normalize_angle_pi(strip["angle_local_rad"] + rotation_rad),
        })
    return {
        "strips": placed,
        "half_width_m": half_width,
        "center_uv_m": center,
        "center_source": center_source,
        "rotation_deg": float(operation.get("rotation_deg", 0.0)),
        "translation_uv_m": translation,
        "n_strips": n_strips,
    }


def point_segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Vectorised port of zigzag::dist_point_segment_2d."""
    segment = end - start
    length2 = float(segment @ segment)
    if length2 <= 1e-30:
        return np.linalg.norm(points - start, axis=1)
    t = np.clip((points - start) @ segment / length2, 0.0, 1.0)
    closest = start + t[:, None] * segment[None, :]
    return np.linalg.norm(points - closest, axis=1)


def capsule_polygon(start: np.ndarray, end: np.ndarray, half_width: float, arc: int = 24) -> np.ndarray:
    """Exact painted footprint of one strip: {p : dist(p, segment) <= half_width}."""
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-15:
        angles = np.linspace(0.0, 2.0 * math.pi, 2 * arc)
        return start + half_width * np.column_stack((np.cos(angles), np.sin(angles)))
    tangent = direction / norm
    normal = np.array([-tangent[1], tangent[0]])
    base = math.atan2(normal[1], normal[0])
    # cap around `end`: from +normal to -normal through +tangent
    end_arc = np.linspace(base, base - math.pi, arc)
    # cap around `start`: from -normal to +normal through -tangent
    start_arc = np.linspace(base + math.pi, base, arc)
    return np.vstack((
        end + half_width * np.column_stack((np.cos(end_arc), np.sin(end_arc))),
        start + half_width * np.column_stack((np.cos(start_arc), np.sin(start_arc))),
    ))


# ---------------------------------------------------------------------------
# case discovery
# ---------------------------------------------------------------------------
def expand_schedule(sequence: dict) -> list:
    """Cycle-by-cycle schedule the solver executed (repeat expanded, disabled skipped)."""
    schedule = []
    for toolpath in sequence["toolpaths"]:
        if not toolpath.get("enabled", True):
            continue
        repeat = int(toolpath.get("repeat", 1))
        for repeat_index in range(1, repeat + 1):
            schedule.append({
                "toolpath_id": toolpath["id"],
                "repeat_index": repeat_index,
                "operation": toolpath["operation"],
            })
    return schedule


def discover_case(case_dir: Path) -> dict:
    result_path = case_dir / "result.json"
    manifest_path = case_dir / "manifest.json"
    sequence_path = case_dir / "sequence.json"
    if not (result_path.is_file() and manifest_path.is_file() and sequence_path.is_file()):
        raise FileNotFoundError(f"{case_dir} is missing result.json/manifest.json/sequence.json")

    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    sequence = json.loads(sequence_path.read_text())

    initial = None
    finals: dict = {}
    mappings: dict = {}
    for path in sorted(case_dir.glob("*.vtp")):
        match = INITIAL_RE.search(path.name)
        if match:
            initial = path
            continue
        match = CYCLE_RE.search(path.name)
        if not match:
            continue
        index = int(match.group(1))
        (finals if match.group(4) == "final" else mappings)[index] = path
    if initial is None:
        raise FileNotFoundError(f"{case_dir}: no cycle_000_initial VTP (reference geometry)")

    schedule = expand_schedule(sequence)
    return {
        "case_dir": case_dir,
        "name": manifest["case"]["name"],
        "tier": manifest["case"].get("tier", manifest.get("family", case_dir.parent.name)),
        "case_spec": manifest["case"],
        "manifest": manifest,
        "sequence": sequence,
        "result": result,
        "initial_vtp": initial,
        "final_vtps": finals,
        "mapping_vtps": mappings,
        "final_vtp": final_vtp(case_dir, result),
        "schedule": schedule,
    }


def completed_cases(study_root: Path, patterns: list) -> list:
    cases = []
    for result_path in sorted(study_root.glob("*/*/result.json")):
        case_dir = result_path.parent
        if patterns and not any(pattern in case_dir.name for pattern in patterns):
            continue
        result = json.loads(result_path.read_text())
        if result.get("return_code") != 0:
            print(f"[skip] {case_dir.name}: return_code={result.get('return_code')}")
            continue
        cases.append(discover_case(case_dir))
    return cases


# ---------------------------------------------------------------------------
# mesh loading / field export
# ---------------------------------------------------------------------------
def load_mesh(path: Path) -> dict:
    data, points, triangles = load_vtp(path)
    material = np.column_stack((point_array(data, "material_u"), point_array(data, "material_v")))
    return {"path": path, "data": data, "points": points, "triangles": triangles, "material": material}


def cell_centroids(values: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    return values[triangles].mean(axis=1)


def displacement_fields(reference: dict, current: dict) -> dict:
    raw = point_array(current["data"], "U_from_cycle0")
    geometric = current["points"] - reference["points"]
    aligned_points, kabsch_rmse, kabsch_max = proper_kabsch(reference["points"], current["points"])
    aligned = aligned_points - reference["points"]
    return {
        "U_raw_m": raw,
        "U_raw_magnitude_m": np.linalg.norm(raw, axis=1),
        "U_aligned_m": aligned,
        "U_aligned_magnitude_m": np.linalg.norm(aligned, axis=1),
        "aligned_points_m": aligned_points,
        "consistency": {
            "max_abs_diff_U_from_cycle0_vs_current_minus_reference_m":
                float(np.abs(raw - geometric).max()),
            "max_abs_diff_Umag_array_vs_norm_m":
                float(np.abs(point_array(current["data"], "Umag_from_cycle0")
                             - np.linalg.norm(raw, axis=1)).max()),
            "rigid_alignment_rmse_m": kabsch_rmse,
            "rigid_alignment_max_deviation_m": kabsch_max,
        },
    }


def write_field_csv(path: Path, reference: dict, current: dict, fields: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    material = current["material"]
    rows = np.column_stack((
        np.arange(current["points"].shape[0]),
        material,
        reference["points"],
        current["points"],
        fields["U_raw_m"],
        fields["U_raw_magnitude_m"],
        fields["U_aligned_m"],
        fields["U_aligned_magnitude_m"],
    ))
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELD_COLUMNS)
        for row in rows:
            writer.writerow(["%d" % int(row[0])] + ["%.12e" % value for value in row[1:]])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def nice_exaggeration(plan_extent: float, out_of_plane: float) -> float:
    if out_of_plane <= 0.0:
        return 1.0
    target = 0.2 * plan_extent / out_of_plane
    for candidate in (1, 2, 5, 10, 20, 50, 100, 200, 500):
        if candidate >= target:
            return float(candidate)
    return 1000.0


def set_true_scale_axes(axis, points_mm: np.ndarray) -> None:
    lower = points_mm.min(axis=0)
    upper = points_mm.max(axis=0)
    centre = 0.5 * (lower + upper)
    half = 0.5 * max(upper - lower)
    axis.set_xlim(centre[0] - half, centre[0] + half)
    axis.set_ylim(centre[1] - half, centre[1] + half)
    axis.set_zlim(centre[2] - half, centre[2] + half)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def set_stretched_axes(axis, points_mm: np.ndarray) -> None:
    """Data-fitted 3D box (used only for explicitly exaggerated views)."""
    lower = points_mm.min(axis=0)
    upper = points_mm.max(axis=0)
    span = np.maximum(upper - lower, 1e-9)
    pad = 0.04 * span
    axis.set_xlim(lower[0] - pad[0], upper[0] + pad[0])
    axis.set_ylim(lower[1] - pad[1], upper[1] + pad[1])
    axis.set_zlim(lower[2] - pad[2], upper[2] + pad[2])
    axis.set_box_aspect((span[0], span[1], 0.55 * max(span[0], span[1])))


def shape_figure(out_path: Path, case: dict, cycle_label: str, reference: dict,
                 current: dict, fields: dict, dpi: int) -> Path:
    points_mm = current["points"] * 1e3
    material_mm = current["material"] * 1e3
    triangles = current["triangles"]
    umag_mm = fields["U_raw_magnitude_m"] * 1e3
    uz_mm = fields["U_raw_m"][:, 2] * 1e3
    aligned_mm = fields["U_aligned_magnitude_m"] * 1e3

    plan_extent = float(max(points_mm[:, 0].ptp(), points_mm[:, 1].ptp()))
    out_of_plane = float(points_mm[:, 2].ptp())
    factor = nice_exaggeration(plan_extent, out_of_plane)

    figure = plt.figure(figsize=(16.0, 11.5), constrained_layout=True)
    figure.suptitle(
        "%s / %s  --  cycle %s\ncurrent shape (deformed coordinates) vs displacement field "
        "U = x_current - x_reference(cycle_000_initial)\n"
        "source cycle_%s vs cycle_000_initial | axes in mm (1 mm = 1e-3 m); CSV export in metres"
        % (case["tier"], case["name"], cycle_label, cycle_label),
        fontsize=12, fontweight="bold")

    axis = figure.add_subplot(2, 2, 1, projection="3d")
    norm = mcolors.Normalize(vmin=float(umag_mm.min()), vmax=float(umag_mm.max()))
    surface = axis.plot_trisurf(points_mm[:, 0], points_mm[:, 1], points_mm[:, 2],
                                triangles=triangles, cmap="viridis", norm=norm,
                                edgecolor="none", antialiased=False, shade=False)
    surface.set_array(umag_mm[triangles].mean(axis=1))
    surface.set_norm(norm)
    set_true_scale_axes(axis, points_mm)
    axis.set_title("(a) current shape, TRUE 1:1:1 physical scale\ncolour = |U_raw| (mm)", fontsize=11)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.view_init(elev=26, azim=-58)
    figure.colorbar(surface, ax=axis, shrink=0.62, pad=0.09, label="|U_raw| (mm)")

    axis = figure.add_subplot(2, 2, 2, projection="3d")
    exaggerated = points_mm.copy()
    z_centre = float(points_mm[:, 2].mean())
    exaggerated[:, 2] = z_centre + factor * (points_mm[:, 2] - z_centre)
    norm_z = mcolors.TwoSlopeNorm(vmin=min(float(uz_mm.min()), -1e-9),
                                  vcenter=0.0,
                                  vmax=max(float(uz_mm.max()), 1e-9))
    surface = axis.plot_trisurf(exaggerated[:, 0], exaggerated[:, 1], exaggerated[:, 2],
                                triangles=triangles, cmap="coolwarm", norm=norm_z,
                                edgecolor="none", antialiased=False, shade=False)
    surface.set_array(uz_mm[triangles].mean(axis=1))
    surface.set_norm(norm_z)
    set_stretched_axes(axis, exaggerated)
    axis.set_title("(b) current shape, vertical exaggeration x%g (NOT physical scale)\n"
                   "colour = U_raw,z (mm); z tick labels are exaggerated" % factor, fontsize=11)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z x%g (mm)" % factor)
    axis.view_init(elev=26, azim=-58)
    figure.colorbar(surface, ax=axis, shrink=0.62, pad=0.09, label="U_raw,z (mm)")

    axis = figure.add_subplot(2, 2, 3)
    contour = axis.tripcolor(material_mm[:, 0], material_mm[:, 1], triangles, uz_mm,
                             cmap="coolwarm", norm=norm_z, shading="gouraud")
    axis.set_aspect("equal")
    axis.set_title("(c) out-of-plane displacement U_raw,z over MATERIAL coordinates\n"
                   "(includes rigid-body motion of the numerical gauge)", fontsize=11)
    axis.set_xlabel("material u (mm)")
    axis.set_ylabel("material v (mm)")
    figure.colorbar(contour, ax=axis, shrink=0.85, label="U_raw,z (mm)")

    axis = figure.add_subplot(2, 2, 4)
    contour = axis.tripcolor(material_mm[:, 0], material_mm[:, 1], triangles, aligned_mm,
                             cmap="magma", shading="gouraud")
    axis.set_aspect("equal")
    axis.set_title("(d) |U| after proper-Kabsch rigid alignment onto the reference\n"
                   "(rigid-body motion removed; shape change only)", fontsize=11)
    axis.set_xlabel("material u (mm)")
    axis.set_ylabel("material v (mm)")
    figure.colorbar(contour, ax=axis, shrink=0.85, label="|U_rigid_aligned| (mm)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


def validate_reconstruction(placement: dict, mesh: dict, hits: np.ndarray) -> dict:
    """Check the reconstructed centrelines against the solver's own mapping arrays.

    The solver stores one hit block per face and per pass (``*_hit_01``,
    ``*_hit_02``, ...) in strip traversal order.  The reconstruction reproduces
    the same ordered blocks, so strip indices, per-face pass counts and material
    growth angles can be compared entry by entry.
    """
    centroids = cell_centroids(mesh["material"], mesh["triangles"])
    strips = placement["strips"]
    distances = np.column_stack([
        point_segment_distance(centroids, strip["a_uv_m"], strip["b_uv_m"])
        for strip in strips
    ])
    covered = distances <= placement["half_width_m"]
    predicted_hits = covered.sum(axis=1)

    faces = centroids.shape[0]
    depth = max(1, int(predicted_hits.max()))
    expected_strip = np.full((faces, depth), -1.0)
    expected_angle = np.zeros((faces, depth))
    rank = np.cumsum(covered, axis=1) - 1
    for column, strip in enumerate(strips):
        rows = np.nonzero(covered[:, column])[0]
        expected_strip[rows, rank[rows, column]] = strip["strip_index"]
        expected_angle[rows, rank[rows, column]] = strip["angle_material_rad"]

    data = mesh["data"]
    angle_error = 0.0
    strip_mismatches = 0
    compared = 0
    blocks = 0
    while True:
        suffix = blocks + 1
        name = "growth_angle_material_hit_%02d" % suffix
        if data.GetCellData().GetArray(name) is None:
            break
        active = cell_array(data, "hit_active_%02d" % suffix) > 0.5
        if suffix <= depth:
            mask = active
            difference = np.abs(cell_array(data, name)[mask] - expected_angle[mask, suffix - 1])
            difference = np.minimum(difference, math.pi - difference)
            if difference.size:
                angle_error = max(angle_error, float(difference.max()))
            strip_mismatches += int(np.count_nonzero(
                cell_array(data, "strip_index_hit_%02d" % suffix)[mask]
                != expected_strip[mask, suffix - 1]))
            compared += int(mask.sum())
        else:
            strip_mismatches += int(active.sum())
        blocks += 1

    hit_faces = hits > 0.5
    return {
        "reconstructed_contact_faces": int((predicted_hits > 0).sum()),
        "solver_contact_faces": int(hit_faces.sum()),
        "contact_face_set_identical": bool(np.array_equal(predicted_hits > 0, hit_faces)),
        "hit_count_identical": bool(np.array_equal(predicted_hits, hits.astype(int))),
        "max_hit_count_difference": int(np.abs(predicted_hits - hits.astype(int)).max()),
        "solver_hit_blocks_in_mapping_vtp": blocks,
        "strip_index_mismatches": strip_mismatches,
        "max_growth_angle_error_rad": angle_error,
        "hit_blocks_compared": compared,
    }


def draw_toolpath(axis, placement: dict, mesh: dict, hits: np.ndarray, bbox: tuple,
                  show_arrows: bool = True, legend: bool = False):
    umin, umax, vmin, vmax = [value * 1e3 for value in bbox]
    axis.add_patch(plt.Rectangle((umin, vmin), umax - umin, vmax - vmin,
                                 fill=False, edgecolor="0.25", linewidth=1.2, zorder=1))

    material_mm = mesh["material"] * 1e3
    triangles = mesh["triangles"]
    contact = hits > 0.5
    collection = None
    if np.any(contact):
        polygons = material_mm[triangles[contact]]
        counts = hits[contact]
        collection = PolyCollection(polygons, array=counts, cmap="YlOrRd",
                                    norm=mcolors.Normalize(vmin=0.5, vmax=max(1.5, counts.max() + 0.5)),
                                    edgecolors="none", zorder=2)
        axis.add_collection(collection)

    footprint = [capsule_polygon(strip["a_uv_m"] * 1e3, strip["b_uv_m"] * 1e3,
                                 placement["half_width_m"] * 1e3)
                 for strip in placement["strips"]]
    axis.add_collection(PolyCollection(footprint, facecolors="none", edgecolors="tab:blue",
                                       linewidths=0.7, linestyles="--", zorder=3))

    for order, strip in enumerate(placement["strips"]):
        start = strip["a_uv_m"] * 1e3
        end = strip["b_uv_m"] * 1e3
        axis.plot([start[0], end[0]], [start[1], end[1]], color="k", linewidth=1.4, zorder=4,
                  label="reconstructed strip centreline" if (legend and order == 0) else None)
        if show_arrows:
            middle = 0.5 * (start + end)
            direction = end - start
            axis.annotate("", xy=middle + 0.18 * direction, xytext=middle - 0.18 * direction,
                          arrowprops=dict(arrowstyle="-|>", color="k", linewidth=1.0), zorder=5)
    first = placement["strips"][0]["a_uv_m"] * 1e3
    axis.plot([first[0]], [first[1]], marker="o", color="lime", markersize=5,
              markeredgecolor="k", zorder=6,
              label="traversal start (strip 0)" if legend else None)

    pad = 0.03 * max(umax - umin, vmax - vmin)
    axis.set_xlim(umin - pad, umax + pad)
    axis.set_ylim(vmin - pad, vmax + pad)
    axis.set_aspect("equal")
    return collection


def toolpath_figure(out_path: Path, case: dict, cycle_index: int, entry: dict,
                    placement: dict, mesh: dict, hits: np.ndarray, validation: dict,
                    bbox: tuple, dpi: int) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(15.0, 7.4), constrained_layout=True)
    operation = entry["operation"]

    collection = draw_toolpath(axes[0], placement, mesh, hits, bbox, legend=True)
    axes[0].set_title("cycle %03d  %s\nexact reconstructed zigzag on the plate (material coordinates)"
                      % (cycle_index, entry["toolpath_id"]), fontsize=12, fontweight="bold")
    axes[0].set_xlabel("material u (mm)")
    axes[0].set_ylabel("material v (mm)")
    axes[0].legend(fontsize=8, loc="upper right")
    if collection is not None:
        figure.colorbar(collection, ax=axes[0], shrink=0.8,
                        label="solver hits_this_cycle per face (mapping VTP)")

    axis = axes[1]
    axis.axis("off")
    lines = [
        "recipe (sequence.json, lengths mm, angles deg):",
        "  lv_mm=%g  alpha_deg=%g  n_strips=%d  width_mm=%g"
        % (operation["lv_mm"], operation["alpha_deg"], operation["n_strips"], operation["width_mm"]),
        "  center_uv_mm=%s  rotation_deg=%g"
        % (operation.get("center_uv_mm", "material bbox centre"), operation.get("rotation_deg", 0.0)),
        "  gtop=%s  gbot=%s  ortho=%s"
        % (operation.get("gtop"), operation.get("gbot"), operation.get("ortho")),
        "  repeat index %d" % entry["repeat_index"],
        "",
        "geometry provenance:",
        "  centrelines rebuilt from sequence.json with the exact port of",
        "  zigzag::buildZigZagStrips + the placement transform of",
        "  zigzag::collectMaterialCoordinateHits (src/libshell/ZigZagGrowth.hpp)",
        "  p_uv = center + R(rotation_deg) p_local + translation",
        "  centre source: %s" % placement["center_source"],
        "  dashed blue outline: analytic painted footprint,",
        "    {p : dist(p, centreline) <= width/2}",
        "",
        "validation against the solver's own mapping output",
        "(%s):" % mesh["path"].name,
        "  faces in contact, solver hits_this_cycle : %d" % validation["solver_contact_faces"],
        "  faces in contact, reconstruction         : %d" % validation["reconstructed_contact_faces"],
        "  identical contact-face set               : %s" % validation["contact_face_set_identical"],
        "  identical per-face hit counts            : %s" % validation["hit_count_identical"],
        "  max |hit-count difference|               : %d" % validation["max_hit_count_difference"],
        "  strip-index mismatches over all passes   : %d" % validation["strip_index_mismatches"],
        "  max growth-angle error                   : %.3e rad (%d face-passes)"
        % (validation["max_growth_angle_error_rad"], validation["hit_blocks_compared"]),
        "",
        "the shaded cells are exported solver face hits, not a contact-",
        "mechanics calculation; the black polyline reconstructs the recipe",
        "in reference material coordinates, not the deformed tool trajectory.",
    ]
    text = "\n".join(textwrap.fill(line, width=76, subsequent_indent="    ")
                     for line in lines)
    axis.text(0.0, 1.0, text, va="top", ha="left", family="monospace", fontsize=9.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


def sequence_overview_figure(out_path: Path, case: dict, cycles: list, dpi: int) -> Path:
    count = len(cycles)
    columns = 5
    rows = int(math.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.1 * columns, 3.5 * rows),
                                constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for axis in axes[count:]:
        axis.axis("off")
    for axis, item in zip(axes, cycles):
        draw_toolpath(axis, item["placement"], item["mesh"], item["hits"], item["bbox"],
                      show_arrows=False)
        axis.set_title("cycle %03d\n%s" % (item["cycle_index"], item["toolpath_id"]), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
    umin, umax, vmin, vmax = cycles[0]["bbox"]
    figure.suptitle("%s / %s -- executed zigzag toolpath sequence on the plate\n"
                    "black: reconstructed centreline of the executed recipe;\n"
                    "shaded: solver per-face contact (hits_this_cycle); "
                    "dashed: analytic strip footprint\n"
                    "thin rectangle: plate outline measured from the mesh material coordinates, "
                    "%.1f x %.1f mm (u in [%.1f, %.1f], v in [%.1f, %.1f] mm)"
                    % (case["tier"], case["name"],
                       (umax - umin) * 1e3, (vmax - vmin) * 1e3,
                       umin * 1e3, umax * 1e3, vmin * 1e3, vmax * 1e3),
                    fontsize=12, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


def coverage_figure(out_path: Path, case: dict, final_mesh: dict, fields: dict, dpi: int) -> Path:
    total = cell_array(final_mesh["data"], "total_hit_count")
    material_mm = final_mesh["material"] * 1e3
    points_mm = final_mesh["points"] * 1e3
    triangles = final_mesh["triangles"]

    figure, axes = plt.subplots(1, 2, figsize=(15.0, 7.0), constrained_layout=True)
    polygons = material_mm[triangles]
    collection = PolyCollection(polygons, array=total, cmap="YlOrRd", edgecolors="none")
    axes[0].add_collection(collection)
    axes[0].set_xlim(material_mm[:, 0].min(), material_mm[:, 0].max())
    axes[0].set_ylim(material_mm[:, 1].min(), material_mm[:, 1].max())
    axes[0].set_aspect("equal")
    axes[0].set_title("cumulative contact map over the whole sequence\n"
                      "solver cell array total_hit_count (number of strip passes per face)",
                      fontsize=11)
    axes[0].set_xlabel("material u (mm)")
    axes[0].set_ylabel("material v (mm)")
    figure.colorbar(collection, ax=axes[0], shrink=0.85, label="total_hit_count (passes)")

    axes[1].remove()
    axis = figure.add_subplot(1, 2, 2, projection="3d")
    surface = axis.plot_trisurf(points_mm[:, 0], points_mm[:, 1], points_mm[:, 2],
                                triangles=triangles, cmap="YlOrRd", edgecolor="none",
                                antialiased=False, shade=False)
    surface.set_array(total)
    surface.set_clim(float(total.min()), float(total.max()))
    set_true_scale_axes(axis, points_mm)
    axis.set_title("same contact map on the final CURRENT SHAPE\ntrue 1:1:1 physical scale",
                   fontsize=11)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.view_init(elev=26, azim=-58)
    figure.colorbar(surface, ax=axis, shrink=0.62, pad=0.09, label="total_hit_count (passes)")

    figure.suptitle("%s / %s -- toolpath coverage, %s"
                    % (case["tier"], case["name"], final_mesh["path"].name),
                    fontsize=13, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# cross-mesh shape / curvature comparison
# ---------------------------------------------------------------------------
def midline_profile(mesh: dict, values: np.ndarray, axis: int, samples: int = 200) -> tuple:
    material_mm = mesh["material"] * 1e3
    triangulation = mtri.Triangulation(material_mm[:, 0], material_mm[:, 1], mesh["triangles"])
    interpolator = mtri.LinearTriInterpolator(triangulation, values)
    lower = material_mm[:, axis].min()
    upper = material_mm[:, axis].max()
    coordinate = np.linspace(lower * 0.999, upper * 0.999, samples)
    if axis == 0:
        sampled = interpolator(coordinate, np.zeros_like(coordinate))
    else:
        sampled = interpolator(np.zeros_like(coordinate), coordinate)
    return coordinate, np.ma.filled(sampled, np.nan)


def material_frame_curvature(material: np.ndarray, out_of_plane: np.ndarray) -> dict:
    """Small-deflection curvature of a height field over the fixed material frame.

    Fits w(u, v) = 0.5 k_uu u^2 + k_uv u v + 0.5 k_vv v^2 + linear + const in the
    persistent material coordinates, so the orientation of the bending axis is
    directly comparable between meshes (the SVD frame of ``quadratic_fit`` is not).
    """
    u = material[:, 0]
    v = material[:, 1]
    design = np.column_stack((0.5 * u * u, u * v, 0.5 * v * v, u, v, np.ones_like(u)))
    coefficients, _, _, _ = np.linalg.lstsq(design, out_of_plane, rcond=None)
    k_uu, k_uv, k_vv = (float(value) for value in coefficients[:3])
    residual = out_of_plane - design @ coefficients
    hessian = np.array([[k_uu, k_uv], [k_uv, k_vv]])
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    dominant = eigenvectors[:, int(np.argmax(np.abs(eigenvalues)))]
    return {
        "k_uu_m_inv": k_uu,
        "k_uv_m_inv": k_uv,
        "k_vv_m_inv": k_vv,
        "dominant_curvature_m_inv": float(eigenvalues[int(np.argmax(np.abs(eigenvalues)))]),
        "dominant_direction_deg_from_u": float(math.degrees(math.atan2(dominant[1], dominant[0])) % 180.0),
        "fit_rmse_m": float(np.sqrt(np.mean(residual**2))),
    }


def mesh_triplet_figure(out_path: Path, cases: list, names: list, dpi: int) -> dict:
    """Three-mesh comparison of the same sequence with IDENTICAL axis and colour limits."""
    lookup = {case["name"]: case for case in cases}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise SystemExit("mesh-triplet cases not available: %s" % ", ".join(missing))

    panels = []
    for name in names:
        case = lookup[name]
        reference = load_mesh(case["initial_vtp"])
        current = load_mesh(case["final_vtp"])
        fields = displacement_fields(reference, current)
        aligned_mm = (fields["aligned_points_m"] - reference["points"]) * 1e3
        fit = quadratic_fit(current["points"])
        panels.append({
            "case": case,
            "mesh": current,
            "aligned_uz_mm": aligned_mm[:, 2],
            "aligned_magnitude_mm": fields["U_aligned_magnitude_m"] * 1e3,
            "aligned_points_mm": fields["aligned_points_m"] * 1e3,
            "fit": fit,
            "material_frame": material_frame_curvature(
                current["material"], fields["aligned_points_m"][:, 2] - reference["points"][:, 2]),
            "source": "%s/%s" % (case["name"], current["path"].name),
        })

    all_points = np.vstack([panel["aligned_points_mm"] for panel in panels])
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    centre = 0.5 * (lower + upper)
    half = 0.5 * float(max(upper - lower))
    uz_limit = float(max(abs(np.concatenate([panel["aligned_uz_mm"] for panel in panels])).max(), 1e-9))
    uz_norm = mcolors.TwoSlopeNorm(vmin=-uz_limit, vcenter=0.0, vmax=uz_limit)

    figure = plt.figure(figsize=(6.0 * len(panels), 12.2), constrained_layout=True)
    for column, panel in enumerate(panels):
        mesh = panel["mesh"]
        points_mm = panel["aligned_points_mm"]
        triangles = mesh["triangles"]
        spec = panel["case"]["case_spec"]

        axis = figure.add_subplot(2, len(panels), column + 1, projection="3d")
        surface = axis.plot_trisurf(points_mm[:, 0], points_mm[:, 1], points_mm[:, 2],
                                    triangles=triangles, cmap="coolwarm", norm=uz_norm,
                                    edgecolor="none", antialiased=False, shade=False)
        surface.set_array(panel["aligned_uz_mm"][triangles].mean(axis=1))
        surface.set_norm(uz_norm)
        axis.set_xlim(centre[0] - half, centre[0] + half)
        axis.set_ylim(centre[1] - half, centre[1] + half)
        axis.set_zlim(centre[2] - half, centre[2] + half)
        axis.set_box_aspect((1.0, 1.0, 1.0))
        axis.set_title("%s\n-res %g (dimensionless mesh control, relArea = 2*Lx*res),"
                       " load step %g\n%d vertices / %d faces\n"
                       "rigid-aligned current shape, TRUE 1:1:1 scale"
                       % (panel["case"]["name"], spec.get("res"), spec.get("step"),
                          mesh["points"].shape[0], triangles.shape[0]), fontsize=10)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_zlabel("z (mm)")
        axis.view_init(elev=26, azim=-58)
        if column == len(panels) - 1:
            figure.colorbar(surface, ax=axis, shrink=0.6, pad=0.1,
                            label="rigid-aligned U_z (mm), shared limits")

        axis = figure.add_subplot(2, len(panels), len(panels) + column + 1)
        material_mm = mesh["material"] * 1e3
        contour = axis.tripcolor(material_mm[:, 0], material_mm[:, 1], triangles,
                                 panel["aligned_uz_mm"], cmap="coolwarm", norm=uz_norm,
                                 shading="gouraud")
        axis.set_aspect("equal")
        axis.set_xlim(centre[0] - half, centre[0] + half)
        axis.set_ylim(centre[1] - half, centre[1] + half)
        peak = float(panel["aligned_uz_mm"].max() - panel["aligned_uz_mm"].min())
        axis.set_title("rigid-aligned U_z over material coordinates\n"
                       "height span = %.3f mm, max |U_aligned| = %.3f mm\n"
                       "SVD-frame kappa = (%.4g, %.4g) 1/m\n"
                       "material frame k_uu = %.4g, k_vv = %.4g 1/m\n"
                       "dominant bending axis %.1f deg from u"
                       % (peak, panel["aligned_magnitude_mm"].max(),
                          panel["fit"]["principal_curvatures_m_inv"][0],
                          panel["fit"]["principal_curvatures_m_inv"][1],
                          panel["material_frame"]["k_uu_m_inv"],
                          panel["material_frame"]["k_vv_m_inv"],
                          panel["material_frame"]["dominant_direction_deg_from_u"]), fontsize=9.5)
        axis.set_xlabel("material u (mm)")
        axis.set_ylabel("material v (mm)")
        if column == len(panels) - 1:
            figure.colorbar(contour, ax=axis, shrink=0.85,
                            label="rigid-aligned U_z (mm), shared limits")
        panel["aligned_height_span_mm"] = peak

    figure.suptitle("mesh comparison of the identical 20-path sequence, final state\n"
                    "identical axis limits and identical colour limits in every panel; "
                    "rigid-body motion removed by proper Kabsch alignment onto the flat reference",
                    fontsize=13, fontweight="bold")
    figure.text(0.005, 0.003, "sources: %s" % ", ".join(panel["source"] for panel in panels),
                fontsize=7, color="0.3")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=dpi)
    plt.close(figure)
    return {
        "figure": out_path,
        "panels": [{
            "case": panel["case"]["name"],
            "res_parameter": panel["case"]["case_spec"].get("res"),
            "step": panel["case"]["case_spec"].get("step"),
            "source_vtp": panel["source"],
            "vertices": int(panel["mesh"]["points"].shape[0]),
            "aligned_height_span_mm": panel["aligned_height_span_mm"],
            "max_aligned_displacement_mm": float(panel["aligned_magnitude_mm"].max()),
            "principal_curvatures_m_inv": panel["fit"]["principal_curvatures_m_inv"],
            "material_frame_curvature": panel["material_frame"],
            "quadratic_fit_r2": panel["fit"]["quadratic_r2"],
        } for panel in panels],
        "shared_uz_limit_mm": uz_limit,
    }


def cross_mesh_report(out_dir: Path, cases: list, dpi: int) -> dict:
    records = []
    profiles = []
    for case in cases:
        reference = load_mesh(case["initial_vtp"])
        current = load_mesh(case["final_vtp"])
        fields = displacement_fields(reference, current)
        area = areas(current["points"], current["triangles"])
        weights = area / area.sum()
        aligned_mesh = current
        fit = quadratic_fit(current["points"])
        curvatures = fit["principal_curvatures_m_inv"]
        record = {
            "tier": case["tier"],
            "case": case["name"],
            "res_parameter": case["case_spec"].get("res"),
            "step": case["case_spec"].get("step"),
            "solver": case["case_spec"].get("solver"),
            "metric_update": case["case_spec"].get("metric"),
            "vertices": int(current["points"].shape[0]),
            "faces": int(current["triangles"].shape[0]),
            "max_U_raw_magnitude_m": float(fields["U_raw_magnitude_m"].max()),
            "max_abs_U_raw_z_m": float(np.abs(fields["U_raw_m"][:, 2]).max()),
            "max_U_rigid_aligned_magnitude_m": float(fields["U_aligned_magnitude_m"].max()),
            "aligned_z_span_m":
                float(np.ptp(fields["aligned_points_m"][:, 2] - reference["points"][:, 2])),
            "rigid_alignment_rmse_m": fields["consistency"]["rigid_alignment_rmse_m"],
            "principal_curvature_min_m_inv": curvatures[0],
            "principal_curvature_max_m_inv": curvatures[1],
            "quadratic_fit_rmse_m": fit["quadratic_rmse_m"],
            "quadratic_fit_r2": fit["quadratic_r2"],
            "material_frame_k_uu_m_inv": None,
            "material_frame_k_vv_m_inv": None,
            "material_frame_k_uv_m_inv": None,
            "material_frame_dominant_curvature_m_inv": None,
            "material_frame_dominant_axis_deg_from_u": None,
            "area_weighted_mean_curvature_m_inv":
                float(np.dot(weights, cell_array(current["data"], "mean"))),
            "area_weighted_gauss_curvature_m_inv2":
                float(np.dot(weights, cell_array(current["data"], "gauss"))),
            "current_area_m2": float(area.sum()),
        }
        aligned_uz = fields["U_aligned_m"][:, 2]
        frame = material_frame_curvature(current["material"], aligned_uz)
        record.update({
            "material_frame_k_uu_m_inv": frame["k_uu_m_inv"],
            "material_frame_k_vv_m_inv": frame["k_vv_m_inv"],
            "material_frame_k_uv_m_inv": frame["k_uv_m_inv"],
            "material_frame_dominant_curvature_m_inv": frame["dominant_curvature_m_inv"],
            "material_frame_dominant_axis_deg_from_u": frame["dominant_direction_deg_from_u"],
        })
        records.append(record)
        profiles.append({
            "case": case["name"],
            "tier": case["tier"],
            "u": midline_profile(aligned_mesh, aligned_uz, 0),
            "v": midline_profile(aligned_mesh, aligned_uz, 1),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "cross_mesh_shape_metrics.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    figure, axes = plt.subplots(2, 2, figsize=(15.0, 10.5), constrained_layout=True)
    joint = [record for record in records if record["tier"] == "joint"]

    axis = axes[0, 0]
    for step in sorted({record["step"] for record in joint}):
        subset = sorted((record for record in joint if record["step"] == step),
                        key=lambda item: item["res_parameter"])
        axis.plot([record["res_parameter"] for record in subset],
                  [record["principal_curvature_max_m_inv"] for record in subset],
                  "o-", label="load step %g, kappa_max" % step)
        axis.plot([record["res_parameter"] for record in subset],
                  [record["principal_curvature_min_m_inv"] for record in subset],
                  "s--", label="load step %g, kappa_min" % step)
    axis.set_xlabel("-res (dimensionless mesh control, relArea = 2*Lx*res)")
    axis.set_ylabel("principal curvature of the quadratic fit (1/m)")
    axis.set_title("(a) cross-mesh curvature of the final current shape\n(joint tier)", fontsize=11)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    for step in sorted({record["step"] for record in joint}):
        subset = sorted((record for record in joint if record["step"] == step),
                        key=lambda item: item["res_parameter"])
        axis.plot([record["res_parameter"] for record in subset],
                  [record["max_U_rigid_aligned_magnitude_m"] * 1e3 for record in subset],
                  "o-", label="load step %g" % step)
    axis.set_xlabel("-res (dimensionless mesh control, relArea = 2*Lx*res)")
    axis.set_ylabel("max |U_rigid_aligned| (mm)")
    axis.set_title("(b) rigid-aligned displacement amplitude vs mesh", fontsize=11)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)

    for axis, key, label in ((axes[1, 0], "u", "material u (mm), v = 0"),
                             (axes[1, 1], "v", "material v (mm), u = 0")):
        for profile in profiles:
            coordinate, values = profile[key]
            axis.plot(coordinate, values * 1e3, linewidth=1.2, label=profile["case"])
        axis.set_xlabel(label)
        axis.set_ylabel("rigid-aligned U_z (mm)")
        axis.set_title("(%s) mid-line shape profile of the final state, all cases"
                       % ("c" if key == "u" else "d"), fontsize=11)
        axis.grid(alpha=0.3)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=7.5,
                  title="mid-line profiles, one line per completed case")

    figure.suptitle("cross-mesh / cross-solver comparison of the final current shape\n"
                    "source: run/equilibrium_continuation_study final VTPs (already completed runs)",
                    fontsize=13, fontweight="bold")
    figure_path = out_dir / "cross_mesh_shape_comparison.png"
    figure.savefig(figure_path, dpi=dpi)
    plt.close(figure)

    joint_records = [record for record in records if record["tier"] == "joint"]
    step_sensitivity = []
    for res in sorted({record["res_parameter"] for record in joint_records}):
        group = sorted((record for record in joint_records if record["res_parameter"] == res),
                       key=lambda item: item["step"])
        spans = [record["aligned_z_span_m"] for record in group]
        axes_deg = [record["material_frame_dominant_axis_deg_from_u"] for record in group]
        reference_span = float(np.mean(spans))
        step_sensitivity.append({
            "res_parameter": res,
            "steps": [record["step"] for record in group],
            "aligned_z_span_m": spans,
            "aligned_z_span_relative_spread": float((max(spans) - min(spans)) / reference_span)
            if reference_span > 0.0 else None,
            "material_frame_dominant_axis_deg_from_u": axes_deg,
            "dominant_axis_spread_deg": float(max(axes_deg) - min(axes_deg)),
        })

    return {"records": records, "csv": csv_path, "figure": figure_path,
            "step_sensitivity": step_sensitivity}


# ---------------------------------------------------------------------------
# per-case driver
# ---------------------------------------------------------------------------
def select_cycles(case: dict, selection: str) -> list:
    available = sorted(case["final_vtps"])
    if selection == "final":
        final_index = max(available) if available else 0
        return [final_index]
    if selection == "all":
        return available
    wanted = {int(token) for token in selection.replace(" ", "").split(",") if token}
    return [index for index in available if index in wanted]


def process_case(case: dict, out_root: Path, cycles: str, dpi: int,
                 toolpaths: bool) -> dict:
    out_dir = out_root / case["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = load_mesh(case["initial_vtp"])
    bbox = (float(reference["material"][:, 0].min()), float(reference["material"][:, 0].max()),
            float(reference["material"][:, 1].min()), float(reference["material"][:, 1].max()))

    provenance = {
        "case": case["name"],
        "tier": case["tier"],
        "case_spec": case["case_spec"],
        "case_dir": rel(case["case_dir"]),
        "solver_binary_sha256": case["manifest"].get("binary_sha256", case["manifest"].get("hashes", {}).get("binary_sha256")),
        "sequence_sha256": case["manifest"].get("sequence_sha256", case["manifest"].get("hashes", {}).get("sequence_sha256")),
        "git_revision": case["manifest"].get("git_revision"),
        "run_host": case["manifest"].get("host"),
        "run_finished_utc": case["result"].get("finished_utc"),
        "solver_command": case["manifest"].get("command"),
        "reference_geometry_vtp": case["initial_vtp"].name,
        "reference_geometry_note":
            "flat plate at z = 0; vertex coordinates coincide with the material coordinates "
            "to within the VTP float precision",
        "plate_material_bbox_m": {"u_min": bbox[0], "u_max": bbox[1],
                                  "v_min": bbox[2], "v_max": bbox[3]},
        "units": {"exported_csv": "metres (SI) for every length column",
                  "figure_axes": "millimetres (1 mm = 1e-3 m)",
                  "curvature": "1/m", "gauss_curvature": "1/m^2",
                  "case_spec.res": "dimensionless mesh control parameter passed as -res; "
                                   "the geometry uses relArea = 2 * Lx * res, so it is NOT a length",
                  "case_spec.step": "dimensionless sequence load-step fraction"},
        "column_documentation": COLUMN_DOC,
        "displacement_definitions": {
            "U_raw": "current shape minus reference shape, solver point array U_from_cycle0; "
                     "contains rigid-body motion because the sequence is solved with a "
                     "numerical rigid-body gauge instead of physical clamps",
            "U_rigid_aligned": "displacement after proper Kabsch (det=+1) alignment of the "
                               "current shape onto the reference shape; rigid-body motion removed",
        },
        "physical_accuracy_note":
            "geometry-only pipeline output; no measured English-wheel data is involved, "
            "so no claim of physical accuracy is made here",
        "exports": [],
        "figures": [],
        "toolpath_validation": [],
    }

    selected = select_cycles(case, cycles)
    print("[%s] cycles selected for field export/figures: %s" % (case["name"], selected))
    for index in selected:
        current = load_mesh(case["final_vtps"][index])
        fields = displacement_fields(reference, current)
        label = "%03d" % index
        csv_path = out_dir / ("displacement_field_cycle_%s.csv" % label)
        write_field_csv(csv_path, reference, current, fields)
        figure_path = shape_figure(out_dir / ("shape_cycle_%s.png" % label), case, label,
                                   reference, current, fields, dpi)
        provenance["exports"].append({
            "cycle_index": index,
            "source_vtp": current["path"].name,
            "csv": rel(csv_path),
            "rows": int(current["points"].shape[0]),
            "columns": FIELD_COLUMNS,
            "consistency_checks_m": fields["consistency"],
            "max_U_raw_magnitude_m": float(fields["U_raw_magnitude_m"].max()),
            "max_U_rigid_aligned_magnitude_m": float(fields["U_aligned_magnitude_m"].max()),
        })
        provenance["figures"].append(rel(figure_path))

    final_index = max(case["final_vtps"])
    final_mesh = load_mesh(case["final_vtps"][final_index])
    final_fields = displacement_fields(reference, final_mesh)
    provenance["figures"].append(rel(coverage_figure(
        out_dir / "coverage_total_hits.png", case, final_mesh, final_fields, dpi)))

    if toolpaths:
        overview = []
        for index in sorted(case["mapping_vtps"]):
            if index > len(case["schedule"]):
                continue
            entry = case["schedule"][index - 1]
            mesh = load_mesh(case["mapping_vtps"][index])
            hits = cell_array(mesh["data"], "hits_this_cycle")
            placement = place_strips(entry["operation"], bbox)
            validation = validate_reconstruction(placement, mesh, hits)
            validation.update({"cycle_index": index, "toolpath_id": entry["toolpath_id"],
                               "mapping_vtp": mesh["path"].name})
            provenance["toolpath_validation"].append(validation)
            figure_path = toolpath_figure(
                out_dir / ("toolpath_cycle_%03d_%s.png" % (index, entry["toolpath_id"])),
                case, index, entry, placement, mesh, hits, validation, bbox, dpi)
            provenance["figures"].append(rel(figure_path))
            overview.append({"cycle_index": index, "toolpath_id": entry["toolpath_id"],
                             "placement": placement, "mesh": mesh, "hits": hits, "bbox": bbox})
        if overview:
            provenance["figures"].append(rel(sequence_overview_figure(
                out_dir / "toolpath_sequence_overview.png", case, overview, dpi)))

    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT,
                        help="study root containing <tier>/<case>/result.json")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--case", action="append", default=[],
                        help="substring filter on case names (repeatable); default: every completed case")
    parser.add_argument("--cycles", default="final",
                        help="'final', 'all', or a comma-separated list of cycle indices")
    parser.add_argument("--toolpath-case", action="append", default=["joint_r0.03_s1"],
                        help="cases that additionally get per-cycle toolpath pictures")
    parser.add_argument("--no-toolpaths", action="store_true")
    parser.add_argument("--no-cross-mesh", action="store_true")
    parser.add_argument("--mesh-triplet", default="joint_r0.06_s0.5,joint_r0.03_s0.5,joint_r0.015_s0.5",
                        help="comma-separated case names for the identical-limits mesh comparison "
                             "figure; empty string disables it")
    parser.add_argument("--dpi", type=int, default=140)
    arguments = parser.parse_args()

    cases = completed_cases(arguments.study_root, arguments.case)
    if not cases:
        raise SystemExit("no completed cases found under %s" % arguments.study_root)

    summaries = []
    for case in cases:
        toolpaths = (not arguments.no_toolpaths) and any(
            token in case["name"] for token in arguments.toolpath_case)
        cycles = "all" if toolpaths and arguments.cycles == "final" else arguments.cycles
        summaries.append(process_case(case, arguments.out_root, cycles, arguments.dpi, toolpaths))

    report = {
        "parameter_notes": {
            "res": "-res is a dimensionless mesh control parameter (geometry uses "
                   "relArea = 2 * Lx * res); it is not a length",
            "step": "load-step fraction of the sequence continuation, dimensionless",
            "plate_extent": "taken from the mesh material coordinates of every case, never "
                            "inferred from the -lx/-ly flags (which are half extents)",
        },
        "cases": [{"case": item["case"],
                   "plate_material_bbox_m": item["plate_material_bbox_m"],
                   "figures": len(item["figures"]),
                   "field_exports": len(item["exports"]),
                   "toolpath_cycles_validated": len(item["toolpath_validation"])}
                  for item in summaries],
    }
    if not arguments.no_cross_mesh:
        cross = cross_mesh_report(arguments.out_root, cases, arguments.dpi)
        report["cross_mesh_csv"] = rel(cross["csv"])
        report["cross_mesh_figure"] = rel(cross["figure"])
        report["load_step_sensitivity"] = cross["step_sensitivity"]
        report["cross_mesh_records"] = cross["records"]
    triplet_names = [name for name in arguments.mesh_triplet.split(",") if name]
    if triplet_names:
        triplet = mesh_triplet_figure(arguments.out_root / "mesh_comparison_three_meshes.png",
                                      cases, triplet_names, arguments.dpi)
        report["mesh_triplet_figure"] = rel(triplet["figure"])
        report["mesh_triplet_panels"] = triplet["panels"]
        report["mesh_triplet_shared_uz_limit_mm"] = triplet["shared_uz_limit_mm"]
    (arguments.out_root / "visualization_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key != "cross_mesh_records"}, indent=2))


if __name__ == "__main__":
    main()
