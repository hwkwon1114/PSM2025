"""Analyze tolerance-by-resolution multi-zigzag response fields."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def fit_global_quadratic(current: np.ndarray) -> dict[str, object]:
    """Fit a macroscopic quadratic graph and return its principal curvatures."""

    x = current[:, 0] - np.mean(current[:, 0])
    y = current[:, 1] - np.mean(current[:, 1])
    z = current[:, 2]
    design = np.column_stack(
        (np.ones(len(x)), x, y, 0.5 * x * x, x * y, 0.5 * y * y)
    )
    coefficients, *_ = np.linalg.lstsq(design, z, rcond=None)
    fitted = design @ coefficients
    residual = z - fitted
    hessian = np.array(
        [
            [coefficients[3], coefficients[4]],
            [coefficients[4], coefficients[5]],
        ]
    )
    principal, directions = np.linalg.eigh(hessian)
    variance = np.sum((z - np.mean(z)) ** 2)
    weaker = max(min(abs(principal[0]), abs(principal[1])), 1e-15)
    return {
        "global_principal_curvature_1": float(principal[0]),
        "global_principal_curvature_2": float(principal[1]),
        "global_gaussian_curvature": float(np.prod(principal)),
        "global_curvature_anisotropy": float(max(abs(principal)) / weaker),
        "quadratic_fit_rmse": float(np.sqrt(np.mean(residual**2))),
        "quadratic_fit_r2": float(1.0 - np.sum(residual**2) / max(variance, 1e-30)),
        "quadratic_coefficients": coefficients,
        "global_principal_directions": directions,
    }


def read_case(directory: Path) -> dict[str, object]:
    with (directory / "result.tsv").open(newline="", encoding="utf-8") as stream:
        result = list(csv.DictReader(stream, delimiter="\t"))[-1]
    final_file = directory / result["final_file"]
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(final_file))
    reader.Update()
    data = reader.GetOutput()
    point_data = data.GetPointData()
    cell_data = data.GetCellData()
    rest = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    current = vtk_to_numpy(point_data.GetArray("X_current")).astype(float)
    u3 = vtk_to_numpy(point_data.GetArray("U3")).astype(float)
    triangles = vtk_to_numpy(data.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype(int)
    gaussian = vtk_to_numpy(cell_data.GetArray("gauss")).astype(float)
    mean = vtk_to_numpy(cell_data.GetArray("mean")).astype(float)
    area = 0.5 * np.linalg.norm(
        np.cross(
            current[triangles[:, 1]] - current[triangles[:, 0]],
            current[triangles[:, 2]] - current[triangles[:, 0]],
        ),
        axis=1,
    )
    discriminant = np.maximum(mean**2 - gaussian, 0.0)
    root = np.sqrt(discriminant)
    k1 = mean + root
    k2 = mean - root
    double_curved = (np.abs(k1) > 0.05) & (np.abs(k2) > 0.05)
    global_fit = fit_global_quadratic(current)
    return {
        "directory": directory,
        "resolution": float(result["resolution"]),
        "tolerance": float(result["tolerance"]),
        "status": int(result["status"]),
        "wall_seconds": float(result["wall_seconds"]),
        "energy": float(result["energy"]),
        "gradient": float(result["gradient"]),
        "faces": len(triangles),
        "rest": rest,
        "current": current,
        "u3": u3,
        "triangles": triangles,
        "gaussian": gaussian,
        "u3_span": float(np.ptp(u3)),
        "mean_abs_gaussian": float(np.sum(area * np.abs(gaussian)) / np.sum(area)),
        "positive_gaussian_area_fraction": float(np.sum(area[gaussian > 0.0]) / np.sum(area)),
        "negative_gaussian_area_fraction": float(np.sum(area[gaussian < 0.0]) / np.sum(area)),
        "double_curvature_area_fraction": float(np.sum(area[double_curved]) / np.sum(area)),
        **global_fit,
    }


def write_summary(cases: list[dict[str, object]], output_root: Path) -> None:
    fields = (
        "resolution",
        "tolerance",
        "faces",
        "wall_seconds",
        "energy",
        "gradient",
        "u3_span",
        "mean_abs_gaussian",
        "positive_gaussian_area_fraction",
        "negative_gaussian_area_fraction",
        "double_curvature_area_fraction",
        "global_principal_curvature_1",
        "global_principal_curvature_2",
        "global_gaussian_curvature",
        "global_curvature_anisotropy",
        "quadratic_fit_rmse",
        "quadratic_fit_r2",
    )
    with (output_root / "multi_zigzag_sensitivity_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: case[field] for field in fields} for case in cases)


def plot_scalar_grid(cases: list[dict[str, object]], output_root: Path) -> None:
    resolutions = sorted({case["resolution"] for case in cases}, reverse=True)
    tolerances = sorted({case["tolerance"] for case in cases}, reverse=True)
    lookup = {(case["resolution"], case["tolerance"]): case for case in cases}
    fig, axes = plt.subplots(len(resolutions), len(tolerances), figsize=(16, 14))
    displacement_norm = colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    for row, resolution in enumerate(resolutions):
        for column, tolerance in enumerate(tolerances):
            axis = axes[row, column]
            case = lookup.get((resolution, tolerance))
            if case is None:
                axis.text(0.5, 0.5, "missing", ha="center", va="center")
                axis.axis("off")
                continue
            current = case["current"]
            u3 = case["u3"]
            midpoint = 0.5 * (float(np.min(u3)) + float(np.max(u3)))
            half_span = max(0.5 * float(np.ptp(u3)), 1e-15)
            normalized_u3 = (u3 - midpoint) / half_span
            axis.tripcolor(
                current[:, 0] * 1000.0,
                current[:, 1] * 1000.0,
                case["triangles"],
                normalized_u3,
                shading="gouraud",
                cmap="coolwarm",
                norm=displacement_norm,
            )
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"span={1000*case['u3_span']:.1f} mm\n"
                f"k=({case['global_principal_curvature_1']:+.2f},"
                f"{case['global_principal_curvature_2']:+.2f}) | {case['wall_seconds']:.0f}s",
                fontsize=8,
            )
            if row == 0:
                axis.text(0.5, 1.24, f"tol {tolerance:.0e}", transform=axis.transAxes, ha="center", fontsize=10)
            if column == 0:
                axis.text(-0.20, 0.5, f"res {resolution:g}\n{case['faces']:,} faces", transform=axis.transAxes, ha="center", va="center", rotation=90, fontsize=10)
    scalar = plt.cm.ScalarMappable(norm=displacement_norm, cmap="coolwarm")
    scalar.set_array([])
    color_axis = fig.add_axes((0.925, 0.17, 0.015, 0.62))
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.set_label("case-normalized vertical displacement")
    fig.suptitle("Multi-zigzag sensitivity sweep", fontsize=16)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.04, top=0.91, wspace=0.18, hspace=0.32)
    fig.savefig(output_root / "multi_zigzag_sensitivity_grid.png", dpi=190)
    plt.close(fig)


def plot_reference_geometry(cases: list[dict[str, object]], output_root: Path) -> None:
    reference = min(cases, key=lambda case: (case["resolution"], case["tolerance"]))
    current = reference["current"] * 1000.0
    gaussian = reference["gaussian"]
    limit = float(np.quantile(np.abs(gaussian), 0.98))
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("coolwarm")
    fig = plt.figure(figsize=(13.5, 5.8))
    shape_axis = fig.add_subplot(1, 2, 1, projection="3d")
    surface = shape_axis.plot_trisurf(
        current[:, 0], current[:, 1], current[:, 2],
        triangles=reference["triangles"], linewidth=0.0, shade=False,
    )
    surface.set_facecolors(cmap(norm(gaussian)))
    shape_axis.set_title("Deformed geometry colored by $K$")
    shape_axis.set_xlabel("x (mm)")
    shape_axis.set_ylabel("y (mm)")
    shape_axis.set_zlabel("z (mm)")
    z_span = max(float(np.ptp(current[:, 2])), 1.0)
    shape_axis.set_box_aspect((254.0, 304.8, 3.0 * z_span))
    shape_axis.view_init(elev=27.0, azim=-58.0)
    curvature_axis = fig.add_subplot(1, 2, 2)
    image = curvature_axis.tripcolor(
        reference["rest"][:, 0] * 1000.0,
        reference["rest"][:, 1] * 1000.0,
        reference["triangles"], gaussian,
        shading="flat", cmap="coolwarm", norm=norm,
    )
    curvature_axis.set_aspect("equal")
    curvature_axis.set_title("Gaussian curvature")
    curvature_axis.set_xlabel("material x (mm)")
    curvature_axis.set_ylabel("material y (mm)")
    fig.colorbar(
        image,
        ax=curvature_axis,
        label="$K$ (m$^{-2}$), clipped at the 98th percentile",
    )
    fig.suptitle(
        f"Multi-zigzag reference | res {reference['resolution']:g} | "
        f"tol {reference['tolerance']:.0e}\n"
        f"global fit k=({reference['global_principal_curvature_1']:+.3f}, "
        f"{reference['global_principal_curvature_2']:+.3f}) m^-1 | "
        f"K={reference['global_gaussian_curvature']:+.3f} m^-2 | "
        f"anisotropy={reference['global_curvature_anisotropy']:.1f}:1 | "
        "3x vertical display exaggeration"
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(output_root / "multi_zigzag_reference_geometry.png", dpi=200)
    plt.close(fig)


def _binned_profile(
    coordinate: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 36,
) -> tuple[np.ndarray, np.ndarray]:
    selected_coordinate = coordinate[mask]
    selected_values = values[mask]
    edges = np.linspace(selected_coordinate.min(), selected_coordinate.max(), n_bins + 1)
    bin_index = np.clip(np.digitize(selected_coordinate, edges) - 1, 0, n_bins - 1)
    centers = []
    means = []
    for index in range(n_bins):
        members = bin_index == index
        if np.any(members):
            centers.append(float(np.mean(selected_coordinate[members])))
            means.append(float(np.mean(selected_values[members])))
    return np.asarray(centers), np.asarray(means)


def _quadratic_section(
    coordinate: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    design = np.column_stack(
        (
            np.ones(np.count_nonzero(mask)),
            coordinate[mask],
            0.5 * coordinate[mask] ** 2,
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, values[mask], rcond=None)
    prediction = design @ coefficients
    variance = np.sum((values[mask] - np.mean(values[mask])) ** 2)
    r2 = 1.0 - np.sum((values[mask] - prediction) ** 2) / max(variance, 1e-30)
    section_x, section_z = _binned_profile(coordinate, values, mask)
    section_z -= coefficients[0] + coefficients[1] * section_x
    fit_x = np.linspace(coordinate[mask].min(), coordinate[mask].max(), 240)
    fit_z = 0.5 * coefficients[2] * fit_x**2
    return section_x, section_z, fit_x, fit_z, float(coefficients[2]), float(r2)


def plot_double_curvature_views(
    cases: list[dict[str, object]], output_root: Path
) -> None:
    """Show the reference surface along both fitted principal directions."""

    reference = min(cases, key=lambda case: (case["resolution"], case["tolerance"]))
    current = reference["current"]
    centered_xy = current[:, :2] - np.mean(current[:, :2], axis=0)
    directions = reference["global_principal_directions"]
    principal_xy = centered_xy @ directions
    coefficients = reference["quadratic_coefficients"]
    plane = (
        coefficients[0]
        + coefficients[1] * centered_xy[:, 0]
        + coefficients[2] * centered_xy[:, 1]
    )
    detrended_z = current[:, 2] - plane
    principal = np.array(
        [
            reference["global_principal_curvature_1"],
            reference["global_principal_curvature_2"],
        ]
    )

    coordinates_mm = 1000.0 * principal_xy
    z_mm = 1000.0 * detrended_z
    limit = float(np.max(np.abs(z_mm)))
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("coolwarm")
    triangles = reference["triangles"]
    face_z = np.mean(z_mm[triangles], axis=1)

    fig = plt.figure(figsize=(15.5, 10.5))
    perspective = fig.add_subplot(2, 2, 1, projection="3d")
    weak_view = fig.add_subplot(2, 2, 2, projection="3d")
    for axis, azimuth, title in (
        (perspective, -55.0, "Perspective view"),
        (weak_view, 0.0, "View exposing the weak-curvature direction"),
    ):
        surface = axis.plot_trisurf(
            coordinates_mm[:, 0],
            coordinates_mm[:, 1],
            z_mm,
            triangles=triangles,
            linewidth=0.03,
            antialiased=True,
            shade=False,
        )
        surface.set_facecolors(cmap(norm(face_z)))
        axis.set_xlabel("principal coordinate 1 (mm)")
        axis.set_ylabel("principal coordinate 2 (mm)")
        axis.set_zlabel("detrended z (mm)")
        axis.set_box_aspect(
            (
                np.ptp(coordinates_mm[:, 0]),
                np.ptp(coordinates_mm[:, 1]),
                3.0 * np.ptp(z_mm),
            )
        )
        axis.view_init(elev=25.0, azim=azimuth)
        axis.set_title(title)

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    color_axis = fig.add_axes((0.92, 0.56, 0.014, 0.27))
    colorbar = fig.colorbar(scalar, cax=color_axis)
    colorbar.set_label("detrended height (mm)")

    strong_section = fig.add_subplot(2, 2, 3)
    weak_section = fig.add_subplot(2, 2, 4)
    section_band = 0.008

    strong_mask = np.abs(principal_xy[:, 1]) <= section_band
    strong_x, strong_z, strong_fit_x, strong_fit_z, strong_k, strong_r2 = (
        _quadratic_section(principal_xy[:, 0], detrended_z, strong_mask)
    )
    strong_section.plot(
        1000.0 * strong_x,
        1000.0 * strong_z,
        color="tab:blue",
        linewidth=2.3,
        label="central binned section",
    )
    strong_section.plot(
        1000.0 * strong_fit_x,
        1000.0 * strong_fit_z,
        "--",
        color="tab:red",
        linewidth=2.0,
        label="sectional quadratic fit",
    )
    strong_section.set_title(
        f"Strong direction: k={strong_k:+.3f} m$^{{-1}}$, fit $R^2$={strong_r2:.2f}"
    )
    strong_section.set_xlabel("principal coordinate 1 (mm)")
    strong_section.set_ylabel("linear trend removed z (mm)")
    strong_section.grid(alpha=0.25)
    strong_section.legend(fontsize=8)

    weak_targets = np.linspace(
        np.quantile(principal_xy[:, 0], 0.12),
        np.quantile(principal_xy[:, 0], 0.88),
        7,
    )
    weak_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(weak_targets)))
    weak_curvatures = []
    for target, line_color in zip(weak_targets, weak_colors):
        mask = np.abs(principal_xy[:, 0] - target) <= 0.006
        weak_x, weak_z, _, _, weak_k, weak_r2 = _quadratic_section(
            principal_xy[:, 1], detrended_z, mask
        )
        weak_curvatures.append(weak_k)
        weak_section.plot(
            1000.0 * weak_x,
            1000.0 * weak_z,
            color=line_color,
            linewidth=1.8,
            label=f"q1={1000*target:+.0f} mm: k={weak_k:+.2f}, R2={weak_r2:.2f}",
        )
    weak_section.axhline(0.0, color="0.4", linewidth=0.8)
    weak_section.set_title(
        "Weak-direction sections are regional: "
        f"k={min(weak_curvatures):+.2f} to {max(weak_curvatures):+.2f} m$^{{-1}}$"
    )
    weak_section.set_xlabel("principal coordinate 2 (mm)")
    weak_section.set_ylabel("linear trend removed z (mm)")
    weak_section.grid(alpha=0.25)
    weak_section.legend(fontsize=7, ncol=2)

    fig.suptitle(
        "Multi-zigzag 3D and sectional-curvature diagnostic\n"
        f"res {reference['resolution']:g}, tol {reference['tolerance']:.0e}; "
        "best-fit plane removed; 3x vertical display exaggeration in 3D panels",
        fontsize=16,
    )
    fig.subplots_adjust(
        left=0.06, right=0.89, bottom=0.07, top=0.90, wspace=0.20, hspace=0.26
    )
    fig.savefig(output_root / "multi_zigzag_double_curvature_3d.png", dpi=210)
    plt.close(fig)

    views = plt.figure(figsize=(15.5, 6.2))
    view_axes = [
        views.add_subplot(1, 2, 1, projection="3d"),
        views.add_subplot(1, 2, 2, projection="3d"),
    ]
    for axis, azimuth, title in (
        (view_axes[0], -55.0, "Perspective view"),
        (
            view_axes[1],
            0.0,
            "Looking along the strong axis: weak curvature is subtle and regional",
        ),
    ):
        surface = axis.plot_trisurf(
            coordinates_mm[:, 0],
            coordinates_mm[:, 1],
            z_mm,
            triangles=triangles,
            linewidth=0.03,
            antialiased=True,
            shade=False,
        )
        surface.set_facecolors(cmap(norm(face_z)))
        axis.set_xlabel("principal coordinate 1 (mm)")
        axis.set_ylabel("principal coordinate 2 (mm)")
        axis.set_zlabel("detrended z (mm)")
        axis.set_box_aspect(
            (
                np.ptp(coordinates_mm[:, 0]),
                np.ptp(coordinates_mm[:, 1]),
                3.0 * np.ptp(z_mm),
            )
        )
        axis.view_init(elev=25.0, azim=azimuth)
        axis.set_title(title, pad=8)
    view_scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    view_scalar.set_array([])
    view_color_axis = views.add_axes((0.92, 0.19, 0.014, 0.58))
    view_colorbar = views.colorbar(view_scalar, cax=view_color_axis)
    view_colorbar.set_label("detrended height (mm)")
    views.suptitle(
        "Multi-zigzag final surface in fitted principal coordinates\n"
        "best-fit plane removed; 3x vertical display exaggeration",
        fontsize=16,
    )
    views.subplots_adjust(left=0.02, right=0.89, bottom=0.06, top=0.84, wspace=0.03)
    views.savefig(output_root / "multi_zigzag_3d_views.png", dpi=220)
    plt.close(views)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", nargs="?", type=Path,
        default=Path("run/multi_zigzag_sensitivity"),
    )
    args = parser.parse_args()
    cases = []
    for directory in sorted(args.root.glob("r*_t*")):
        if (directory / "result.tsv").exists():
            try:
                cases.append(read_case(directory))
            except (ValueError, FileNotFoundError):
                pass
    if not cases:
        raise SystemExit(f"No completed cases found under {args.root}")
    write_summary(cases, args.root)
    plot_scalar_grid(cases, args.root)
    plot_reference_geometry(cases, args.root)
    plot_double_curvature_views(cases, args.root)
    print(f"Analyzed {len(cases)} completed cases")


if __name__ == "__main__":
    main()
