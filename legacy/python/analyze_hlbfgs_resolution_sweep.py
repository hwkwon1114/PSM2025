"""Evaluate coarse HLBFGS meshes against the archived Putong 0.01 m field."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib import cm, colors
import numpy as np
from scipy.spatial import cKDTree
import vtk
from vtk.util.numpy_support import vtk_to_numpy


SUMMARY_NAME = "bilayer_zigzag_sequence_summary.csv"


def _last_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[-1]


def _wall_seconds(path: Path) -> float:
    if not path.exists():
        return float("nan")
    match = re.search(r"wall_seconds=([0-9.]+)", path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else float("nan")


def _solver_report(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    hlbfgs = re.findall(r"final eps = ([0-9.eE+-]+)", text)
    return {
        "hlbfgs_final_gradient": float(hlbfgs[-1]) if hlbfgs else float("nan"),
    }


def _resolution(path: Path) -> float:
    log = path / "argumentparser.log"
    match = re.search(r"(?:^| )-res ([0-9.eE+-]+)", log.read_text(encoding="utf-8"))
    if not match:
        raise ValueError(f"Cannot read resolution from {log}")
    return float(match.group(1))


def _read_case(directory: Path) -> dict[str, object]:
    summary = _last_row(directory / SUMMARY_NAME)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(directory / summary["final_file"]))
    reader.Update()
    data = reader.GetOutput()
    point_data = data.GetPointData()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    u = vtk_to_numpy(point_data.GetArray("material_u")).astype(float)
    v = vtk_to_numpy(point_data.GetArray("material_v")).astype(float)
    u3 = vtk_to_numpy(point_data.GetArray("U3_from_cycle0")).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    mean_curvature = vtk_to_numpy(data.GetCellData().GetArray("mean")).astype(float)
    edges = np.vstack(
        (
            triangles[:, (0, 1)],
            triangles[:, (1, 2)],
            triangles[:, (2, 0)],
        )
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    material_xy = np.column_stack((u, v))
    median_edge = float(
        np.median(
            np.linalg.norm(
                material_xy[edges[:, 1]] - material_xy[edges[:, 0]], axis=1
            )
        )
    )
    cross = np.cross(
        points[triangles[:, 1]] - points[triangles[:, 0]],
        points[triangles[:, 2]] - points[triangles[:, 0]],
    )
    area = 0.5 * np.linalg.norm(cross, axis=1)
    solver_report = _solver_report(directory / "run.log")
    return {
        "directory": directory,
        "resolution_m": _resolution(directory),
        "faces": len(triangles),
        "median_edge_m": median_edge,
        "elements_across_10mm_band": 0.010 / median_edge,
        "u": u,
        "v": v,
        "u3": u3,
        "points": points,
        "triangles": triangles,
        "energy": float(summary["total_energy"]),
        "u3_span_m": float(np.ptp(u3)),
        "abs_mean_curvature_integral": float(np.sum(area * np.abs(mean_curvature))),
        "signed_mean_curvature_integral": float(np.sum(area * mean_curvature)),
        "wall_seconds": _wall_seconds(directory / "run.log"),
        **solver_report,
    }


def _remove_plane(u: np.ndarray, v: np.ndarray, field: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(u)), u, v))
    coefficients, *_ = np.linalg.lstsq(design, field, rcond=None)
    return field - design @ coefficients


def _interpolate(case: dict, target_u: np.ndarray, target_v: np.ndarray) -> np.ndarray:
    triangulation = mtri.Triangulation(case["u"], case["v"], case["triangles"])
    interpolator = mtri.LinearTriInterpolator(triangulation, case["u3"])
    values = np.asarray(interpolator(target_u, target_v).filled(np.nan), dtype=float)
    missing = ~np.isfinite(values)
    if np.any(missing):
        tree = cKDTree(np.column_stack((case["u"], case["v"])))
        _, indices = tree.query(np.column_stack((target_u[missing], target_v[missing])))
        values[missing] = case["u3"][indices]
    return values


def _local_roughness(case: dict) -> float:
    neighbors = [set() for _ in range(len(case["u3"]))]
    for left, middle, right in case["triangles"]:
        neighbors[left].update((middle, right))
        neighbors[middle].update((left, right))
        neighbors[right].update((left, middle))
    residual = np.asarray(
        [
            case["u3"][index] - np.mean(case["u3"][list(adjacent)])
            if adjacent
            else 0.0
            for index, adjacent in enumerate(neighbors)
        ]
    )
    return float(np.sqrt(np.mean(residual**2)) / max(case["u3_span_m"], 1e-15))


def evaluate(cases: list[dict], reference: dict) -> list[dict[str, object]]:
    reference_field = _remove_plane(reference["u"], reference["v"], reference["u3"])
    reference_rms = float(np.sqrt(np.mean(reference_field**2)))
    reference_roughness = _local_roughness(reference)
    rows = []
    for case in cases:
        interpolated = _interpolate(case, reference["u"], reference["v"])
        candidate_field = _remove_plane(reference["u"], reference["v"], interpolated)
        direct = np.sqrt(np.mean((candidate_field - reference_field) ** 2))
        reflected = np.sqrt(np.mean((-candidate_field - reference_field) ** 2))
        field_error = float(min(direct, reflected) / max(reference_rms, 1e-15))
        energy_error = abs(case["energy"] / reference["energy"] - 1.0)
        span_error = abs(case["u3_span_m"] / reference["u3_span_m"] - 1.0)
        curvature_error = abs(
            case["abs_mean_curvature_integral"]
            / reference["abs_mean_curvature_integral"]
            - 1.0
        )
        raw_roughness_ratio = _local_roughness(case) / max(reference_roughness, 1e-15)
        roughness_ratio = raw_roughness_ratio * (
            reference["median_edge_m"] / case["median_edge_m"]
        ) ** 2
        deep_basin = case["u3_span_m"] >= 0.010
        accepted = (
            deep_basin
            and field_error <= 0.15
            and energy_error <= 0.10
            and span_error <= 0.15
            and curvature_error <= 0.15
            and roughness_ratio <= 2.0
        )
        rows.append(
            {
                "resolution_m": case["resolution_m"],
                "faces": case["faces"],
                "median_edge_m": case["median_edge_m"],
                "elements_across_10mm_band": case["elements_across_10mm_band"],
                "wall_seconds": case["wall_seconds"],
                "hlbfgs_final_gradient": case["hlbfgs_final_gradient"],
                "u3_span_m": case["u3_span_m"],
                "energy": case["energy"],
                "abs_mean_curvature_integral": case["abs_mean_curvature_integral"],
                "field_nrmse": field_error,
                "u3_span_relative_error": span_error,
                "energy_relative_error": energy_error,
                "curvature_relative_error": curvature_error,
                "mesh_scaled_roughness_ratio": roughness_ratio,
                "deep_basin": deep_basin,
                "accepted_15pct": accepted,
            }
        )
    return sorted(rows, key=lambda row: row["resolution_m"], reverse=True)


def write_outputs(rows: list[dict[str, object]], output_directory: Path) -> None:
    csv_path = output_directory / "hlbfgs_resolution_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    ordered = sorted(rows, key=lambda row: row["resolution_m"])
    resolution = np.asarray([row["resolution_m"] for row in ordered])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(resolution, [row["field_nrmse"] for row in ordered], "-o")
    axes[0].axhline(0.15, color="tab:red", ls="--", label="15% threshold")
    axes[0].set_ylabel("full-field $u_3$ NRMSE")
    axes[0].legend()
    axes[1].plot(resolution, [row["u3_span_relative_error"] for row in ordered], "-o", label="$u_3$ span")
    axes[1].plot(resolution, [row["energy_relative_error"] for row in ordered], "-s", label="energy")
    axes[1].plot(resolution, [row["curvature_relative_error"] for row in ordered], "-^", label="$\\int |H|dA$")
    axes[1].axhline(0.15, color="tab:red", ls="--")
    axes[1].set_ylabel("relative error")
    axes[1].legend()
    axes[2].plot(resolution, [row["wall_seconds"] for row in ordered], "-o")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("wall time (s)")
    for axis in axes:
        axis.set_xlabel("mesh resolution parameter (m)")
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    fig.suptitle("HLBFGS Putong resolution sweep against archived res=0.01 reference")
    fig.tight_layout()
    fig.savefig(output_directory / "hlbfgs_resolution_convergence.png", dpi=190)
    plt.close(fig)

    accepted = [row for row in rows if row["accepted_15pct"]]
    fastest = min(accepted, key=lambda row: row["wall_seconds"]) if accepted else None
    lines = [
        "# HLBFGS coarse-resolution sweep",
        "",
        "Reference: archived Putong one-step HLBFGS solution at resolution 0.01 m,",
        "thickness 0.6 mm, and tolerance 1e-12.",
        "",
        "A case passes the provisional field criterion when it reaches the deep basin,",
        "has full-field U3 NRMSE <= 15%, energy/span/integrated-curvature errors <= 15%",
        "(energy is held to 10%), and edge-length-scaled local roughness <= 2x the reference.",
        "",
    ]
    if fastest:
        lines.append(
            f"Fastest passing case: resolution {fastest['resolution_m']:.4g} m, "
            f"{fastest['faces']} faces, {fastest['wall_seconds']:.1f} s."
        )
    else:
        lines.append("No tested case passed every provisional threshold.")
    lines.extend(["", "See `hlbfgs_resolution_summary.csv` for all metrics.", ""])
    (output_directory / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_geometries(
    cases: list[dict[str, object]],
    reference: dict[str, object],
    rows: list[dict[str, object]],
    output_directory: Path,
) -> None:
    by_resolution = {round(case["resolution_m"], 6): case for case in cases}
    diagnostics = {round(row["resolution_m"], 6): row for row in rows}
    selected = [
        ("Reference", reference, None),
        ("Resolution 0.080", by_resolution[0.08], diagnostics[0.08]),
        ("Resolution 0.060", by_resolution[0.06], diagnostics[0.06]),
        ("Resolution 0.050", by_resolution[0.05], diagnostics[0.05]),
        ("Resolution 0.040", by_resolution[0.04], diagnostics[0.04]),
        ("Resolution 0.035", by_resolution[0.035], diagnostics[0.035]),
        ("Resolution 0.030", by_resolution[0.03], diagnostics[0.03]),
        ("Resolution 0.020", by_resolution[0.02], diagnostics[0.02]),
    ]
    all_z = np.concatenate([entry[1]["points"][:, 2] for entry in selected]) * 1000.0
    limit = float(np.max(np.abs(all_z)))
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("coolwarm")
    fig = plt.figure(figsize=(18, 9.2))
    for index, (label, case, diagnostic) in enumerate(selected):
        axis = fig.add_subplot(2, 4, index + 1, projection="3d")
        points_mm = 1000.0 * case["points"]
        axis.plot_trisurf(
            points_mm[:, 0],
            points_mm[:, 1],
            points_mm[:, 2],
            triangles=case["triangles"],
            cmap=cmap,
            norm=norm,
            linewidth=0.0,
            antialiased=True,
            shade=False,
        )
        if diagnostic is None:
            subtitle = f"Resolution 0.010 | {case['faces']:,} faces"
        else:
            verdict = "PASS" if diagnostic["accepted_15pct"] else "FAIL"
            subtitle = (
                f"{diagnostic['faces']:,} faces | field error {100*diagnostic['field_nrmse']:.1f}%\n"
                f"{verdict} | {diagnostic['wall_seconds']:.1f} s"
            )
        axis.set_title(f"{label}\n{subtitle}", fontsize=10, pad=5)
        axis.set_xlim(-135.0, 135.0)
        axis.set_ylim(-160.0, 160.0)
        axis.set_zlim(-limit, limit)
        axis.set_box_aspect((270.0, 320.0, 2.0 * limit))
        axis.view_init(elev=25.0, azim=-58.0)
        axis.set_xticks((-100.0, 0.0, 100.0))
        axis.set_yticks((-150.0, 0.0, 150.0))
        axis.set_zticks((-20.0, 0.0, 20.0))
        axis.tick_params(labelsize=7, pad=0)
        if index >= 4:
            axis.set_xlabel("x (mm)", fontsize=8, labelpad=1)
            axis.set_ylabel("y (mm)", fontsize=8, labelpad=1)
        if index % 4 == 0:
            axis.set_zlabel("z (mm)", fontsize=8, labelpad=1)
    scalar = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    color_axis = fig.add_axes((0.30, 0.035, 0.40, 0.018))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label("deformed z coordinate (mm)")
    fig.suptitle(
        "HLBFGS mesh-resolution sweep",
        fontsize=16,
        y=0.98,
    )
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.88, wspace=0.02, hspace=0.12)
    path = output_directory / "hlbfgs_resolution_geometries.png"
    fig.savefig(path, dpi=210, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sweep",
        nargs="?",
        type=Path,
        default=Path("run/hlbfgs_resolution_sweep"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("run/zigzag_production/pt_1step"),
    )
    args = parser.parse_args()
    reference = _read_case(args.reference)
    cases = [
        _read_case(directory)
        for directory in args.sweep.iterdir()
        if directory.is_dir() and (directory / SUMMARY_NAME).exists()
    ]
    if not cases:
        raise SystemExit(f"No completed cases in {args.sweep}")
    rows = evaluate(cases, reference)
    write_outputs(rows, args.sweep)
    plot_geometries(cases, reference, rows, args.sweep)
    print(f"Analyzed {len(rows)} HLBFGS cases")


if __name__ == "__main__":
    main()
