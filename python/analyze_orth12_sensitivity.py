"""Analyze the 12-cycle orthogonal zigzag resolution/tolerance sweep."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, tri
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

RESOLUTIONS = (0.03, 0.025, 0.02, 0.015, 0.01)
TOLERANCES = (1e-10, 3e-11, 1e-11, 3e-12, 1e-12)


def fit_global_quadratic(current: np.ndarray) -> dict[str, float]:
    """Fit z(x,y) and return the fitted principal curvatures."""

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
        [[coefficients[3], coefficients[4]], [coefficients[4], coefficients[5]]]
    )
    principal = np.linalg.eigvalsh(hessian)
    weaker = max(float(np.min(np.abs(principal))), 1e-15)
    variance = float(np.sum((z - np.mean(z)) ** 2))
    return {
        "global_principal_curvature_1_m-1": float(principal[0]),
        "global_principal_curvature_2_m-1": float(principal[1]),
        "global_gaussian_curvature_m-2": float(np.prod(principal)),
        "global_curvature_anisotropy": float(np.max(np.abs(principal)) / weaker),
        "quadratic_fit_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "quadratic_fit_r2": float(
            1.0 - np.sum(residual**2) / max(variance, 1e-30)
        ),
    }


def optional_float(value: str) -> float:
    return float(value) if value not in ("", "-") else float("nan")


def read_case(case_dir: Path) -> dict[str, object] | None:
    result_path = case_dir / "result.tsv"
    if not result_path.is_file():
        return None
    with result_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        return None
    result = rows[-1]
    if int(result["status"]) != 0 or not result["verdict"].startswith("MINIMUM_"):
        return None
    final_path = case_dir / result["final_file"]
    if not final_path.is_file():
        return None

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(final_path))
    reader.Update()
    data = reader.GetOutput()
    current = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    triangles = (
        vtk_to_numpy(data.GetPolys().GetData())
        .reshape(-1, 4)[:, 1:]
        .astype(int)
    )
    cell_data = data.GetCellData()
    gaussian_array = cell_data.GetArray("gauss")
    mean_array = cell_data.GetArray("mean")
    if gaussian_array is None or mean_array is None:
        raise RuntimeError(f"Missing gauss/mean fields in {final_path}")
    gaussian = vtk_to_numpy(gaussian_array).astype(float)
    mean = vtk_to_numpy(mean_array).astype(float)
    area = 0.5 * np.linalg.norm(
        np.cross(
            current[triangles[:, 1]] - current[triangles[:, 0]],
            current[triangles[:, 2]] - current[triangles[:, 0]],
        ),
        axis=1,
    )
    root = np.sqrt(np.maximum(mean**2 - gaussian, 0.0))
    local_k1 = mean + root
    local_k2 = mean - root
    double_curved = (np.abs(local_k1) > 0.05) & (np.abs(local_k2) > 0.05)
    total_area = float(np.sum(area))

    return {
        "case": case_dir.name,
        "task_id": int(result["task_id"]),
        "resolution": float(result["resolution"]),
        "tolerance": float(result["tolerance"]),
        "faces": len(triangles),
        "wall_seconds": float(result["wall_seconds"]),
        "energy": optional_float(result["energy"]),
        "gradient": optional_float(result["gradient"]),
        "lambda_min": optional_float(result["lambda_min"]),
        "verdict": result["verdict"],
        "final_file": result["final_file"],
        "z_span_m": float(np.ptp(current[:, 2])),
        "mean_abs_gaussian_curvature_m-2": float(
            np.sum(area * np.abs(gaussian)) / total_area
        ),
        "positive_gaussian_area_fraction": float(
            np.sum(area[gaussian > 0.0]) / total_area
        ),
        "negative_gaussian_area_fraction": float(
            np.sum(area[gaussian < 0.0]) / total_area
        ),
        "double_curvature_area_fraction": float(
            np.sum(area[double_curved]) / total_area
        ),
        **fit_global_quadratic(current),
    }


def write_summary(cases: list[dict[str, object]], output_root: Path) -> None:
    fields = tuple(cases[0])
    with (output_root / "orth12_sensitivity_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cases)


def metric_grid(cases: list[dict[str, object]], field: str, scale: float = 1.0) -> np.ndarray:
    lookup = {
        (case["resolution"], case["tolerance"]): scale * float(case[field])
        for case in cases
    }
    return np.array(
        [
            [lookup.get((resolution, tolerance), np.nan) for tolerance in TOLERANCES]
            for resolution in RESOLUTIONS
        ]
    )


def plot_sensitivity_grid(cases: list[dict[str, object]], output_root: Path) -> None:
    panels = (
        ("z_span_m", 1000.0, "z span (mm)", False),
        ("global_gaussian_curvature_m-2", 1.0, "global K (m$^{-2}$)", True),
        ("global_principal_curvature_1_m-1", 1.0, "global k1 (m$^{-1}$)", True),
        ("global_principal_curvature_2_m-1", 1.0, "global k2 (m$^{-1}$)", True),
        ("global_curvature_anisotropy", 1.0, "anisotropy", False),
        ("quadratic_fit_r2", 1.0, "quadratic fit $R^2$", False),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    for axis, (field, scale, title, diverging) in zip(axes.flat, panels):
        values = metric_grid(cases, field, scale)
        masked = np.ma.masked_invalid(values)
        cmap = plt.get_cmap("coolwarm" if diverging else "viridis").copy()
        cmap.set_bad("0.88")
        norm = None
        finite = values[np.isfinite(values)]
        if diverging and len(finite) and np.min(finite) < 0.0 < np.max(finite):
            norm = colors.TwoSlopeNorm(
                vmin=float(np.min(finite)), vcenter=0.0, vmax=float(np.max(finite))
            )
        image = axis.imshow(masked, aspect="auto", cmap=cmap, norm=norm)
        axis.set_xticks(range(len(TOLERANCES)), [f"{value:.0e}" for value in TOLERANCES])
        axis.set_yticks(range(len(RESOLUTIONS)), [f"{value:g}" for value in RESOLUTIONS])
        axis.set_xlabel("tolerance")
        axis.set_ylabel("resolution")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.84)
    figure.suptitle("12-cycle orthogonal opposite-side sensitivity", fontsize=16)
    figure.savefig(output_root / "orth12_sensitivity_grid.png", dpi=210)
    plt.close(figure)


def plot_convergence(cases: list[dict[str, object]], output_root: Path) -> None:
    reference = min(cases, key=lambda case: (case["resolution"], case["tolerance"]))
    ref_z = max(abs(float(reference["z_span_m"])), 1e-15)
    ref_k = max(abs(float(reference["global_gaussian_curvature_m-2"])), 1e-15)
    ref_pair = np.array(
        [
            reference["global_principal_curvature_1_m-1"],
            reference["global_principal_curvature_2_m-1"],
        ],
        dtype=float,
    )
    ref_pair_norm = max(float(np.linalg.norm(ref_pair)), 1e-15)
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharex=True)
    for tolerance in TOLERANCES:
        selected = sorted(
            (case for case in cases if case["tolerance"] == tolerance),
            key=lambda case: case["resolution"],
        )
        if not selected:
            continue
        resolution = [case["resolution"] for case in selected]
        z_error = [abs(float(case["z_span_m"]) - float(reference["z_span_m"])) / ref_z for case in selected]
        k_error = [abs(float(case["global_gaussian_curvature_m-2"]) - float(reference["global_gaussian_curvature_m-2"])) / ref_k for case in selected]
        pair_error = [
            float(
                np.linalg.norm(
                    np.array(
                        [
                            case["global_principal_curvature_1_m-1"],
                            case["global_principal_curvature_2_m-1"],
                        ],
                        dtype=float,
                    )
                    - ref_pair
                )
                / ref_pair_norm
            )
            for case in selected
        ]
        label = f"tol {tolerance:.0e}"
        axes[0].plot(resolution, z_error, "o-", label=label)
        axes[1].plot(resolution, k_error, "o-", label=label)
        axes[2].plot(resolution, pair_error, "o-", label=label)
    for axis, title in zip(
        axes,
        ("relative z-span error", "relative global-K error", "relative principal-pair error"),
    ):
        axis.set_xlabel("resolution (smaller is finer)")
        axis.set_ylabel("relative error")
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    axes[-1].legend(fontsize=8)
    figure.suptitle(
        f"Convergence relative to res={reference['resolution']:g}, tol={reference['tolerance']:.0e}"
    )
    figure.tight_layout()
    figure.savefig(output_root / "orth12_sensitivity_convergence.png", dpi=210)
    plt.close(figure)


def load_deformations(
    cases: list[dict[str, object]], output_root: Path
) -> dict[tuple[float, float], dict[str, np.ndarray]]:
    """Load actual final geometry and cycle-0 displacement fields."""

    deformations = {}
    for case in cases:
        final_path = output_root / str(case["case"]) / str(case["final_file"])
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(final_path))
        reader.Update()
        data = reader.GetOutput()
        point_data = data.GetPointData()
        material_u = point_data.GetArray("material_u")
        material_v = point_data.GetArray("material_v")
        displacement_u3 = point_data.GetArray("U3_from_cycle0")
        if material_u is None or material_v is None or displacement_u3 is None:
            raise RuntimeError(f"Missing material coordinates or U3 in {final_path}")
        current_mm = 1000.0 * vtk_to_numpy(data.GetPoints().GetData()).astype(float)
        current_mm -= np.mean(current_mm, axis=0)
        triangles = (
            vtk_to_numpy(data.GetPolys().GetData())
            .reshape(-1, 4)[:, 1:]
            .astype(int)
        )
        deformations[(float(case["resolution"]), float(case["tolerance"]))] = {
            "current_mm": current_mm,
            "material_u_mm": 1000.0 * vtk_to_numpy(material_u).astype(float),
            "material_v_mm": 1000.0 * vtk_to_numpy(material_v).astype(float),
            "u3_mm": 1000.0 * vtk_to_numpy(displacement_u3).astype(float),
            "triangles": triangles,
        }
    return deformations


def deformation_norm(
    deformations: dict[tuple[float, float], dict[str, np.ndarray]]
) -> colors.TwoSlopeNorm:
    maximum = max(
        float(np.max(np.abs(deformation["u3_mm"])))
        for deformation in deformations.values()
    )
    return colors.TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum)


def plot_u3_field_gallery(
    cases: list[dict[str, object]],
    deformations: dict[tuple[float, float], dict[str, np.ndarray]],
    output_root: Path,
) -> None:
    """Plot U3 over material coordinates for every sweep case."""

    norm = deformation_norm(deformations)
    levels = np.linspace(norm.vmin, norm.vmax, 25)
    figure, axes = plt.subplots(
        len(RESOLUTIONS),
        len(TOLERANCES),
        figsize=(16.5, 18.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    contour = None
    for row, resolution in enumerate(RESOLUTIONS):
        for column, tolerance in enumerate(TOLERANCES):
            axis = axes[row, column]
            deformation = deformations.get((resolution, tolerance))
            if deformation is None:
                axis.set_axis_off()
                continue
            triangulation = tri.Triangulation(
                deformation["material_u_mm"],
                deformation["material_v_mm"],
                deformation["triangles"],
            )
            contour = axis.tricontourf(
                triangulation,
                deformation["u3_mm"],
                levels=levels,
                cmap="coolwarm",
                norm=norm,
                extend="both",
            )
            axis.tricontour(
                triangulation,
                deformation["u3_mm"],
                levels=[0.0],
                colors="0.2",
                linewidths=0.45,
            )
            axis.set_aspect("equal")
            axis.set_xticks(())
            axis.set_yticks(())
            span = float(np.ptp(deformation["u3_mm"]))
            axis.text(
                0.03,
                0.04,
                f"span {span:.1f} mm",
                transform=axis.transAxes,
                fontsize=8,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65},
            )
            if row == 0:
                axis.set_title(f"tol {tolerance:.0e}")
            if column == 0:
                axis.set_ylabel(f"res {resolution:g}", fontsize=11)
    if contour is not None:
        colorbar = figure.colorbar(
            contour, ax=axes.ravel().tolist(), location="right", shrink=0.76, pad=0.02
        )
        colorbar.set_label("$U_3$ from cycle 0 (mm)")
    figure.suptitle(
        "12-cycle out-of-plane deformation field\n"
        "material coordinates; shared color scale; black line is $U_3=0$",
        fontsize=17,
    )
    figure.savefig(output_root / "orth12_u3_deformation_fields.png", dpi=210)
    plt.close(figure)


def actual_geometry_limits(
    deformations: dict[tuple[float, float], dict[str, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.min(
        np.vstack(
            [np.min(deformation["current_mm"], axis=0) for deformation in deformations.values()]
        ),
        axis=0,
    )
    maximum = np.max(
        np.vstack(
            [np.max(deformation["current_mm"], axis=0) for deformation in deformations.values()]
        ),
        axis=0,
    )
    return minimum, maximum


def draw_deformed_surface(
    axis,
    deformation: dict[str, np.ndarray],
    norm: colors.TwoSlopeNorm,
    limits: tuple[np.ndarray, np.ndarray],
) -> None:
    current = deformation["current_mm"]
    triangles = deformation["triangles"]
    face_u3 = np.mean(deformation["u3_mm"][triangles], axis=1)
    surface = axis.plot_trisurf(
        current[:, 0],
        current[:, 1],
        current[:, 2],
        triangles=triangles,
        cmap="coolwarm",
        norm=norm,
        linewidth=0.015,
        antialiased=True,
        shade=False,
    )
    surface.set_array(face_u3)
    minimum, maximum = limits
    axis.set_xlim(minimum[0], maximum[0])
    axis.set_ylim(minimum[1], maximum[1])
    axis.set_zlim(minimum[2], maximum[2])
    extent = np.maximum(maximum - minimum, 1e-9)
    axis.set_box_aspect(extent)
    axis.view_init(elev=24.0, azim=-55.0)
    axis.set_xticks(())
    axis.set_yticks(())
    axis.set_zticks(())
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_zlabel("")


def plot_deformed_surface_gallery(
    deformations: dict[tuple[float, float], dict[str, np.ndarray]],
    output_root: Path,
) -> None:
    """Plot the actual centered final geometry for every sweep case."""

    norm = deformation_norm(deformations)
    limits = actual_geometry_limits(deformations)
    figure = plt.figure(figsize=(18.5, 18.5))
    axes = np.empty((len(RESOLUTIONS), len(TOLERANCES)), dtype=object)
    for row, resolution in enumerate(RESOLUTIONS):
        for column, tolerance in enumerate(TOLERANCES):
            axis = figure.add_subplot(
                len(RESOLUTIONS), len(TOLERANCES), row * len(TOLERANCES) + column + 1,
                projection="3d",
            )
            axes[row, column] = axis
            deformation = deformations.get((resolution, tolerance))
            if deformation is None:
                axis.set_axis_off()
                continue
            draw_deformed_surface(axis, deformation, norm, limits)
            if row == 0:
                axis.set_title(f"tol {tolerance:.0e}", pad=1)
            if column == 0:
                axis.text2D(
                    -0.10,
                    0.50,
                    f"res {resolution:g}",
                    transform=axis.transAxes,
                    rotation=90,
                    va="center",
                    fontsize=11,
                )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    scalar.set_array([])
    color_axis = figure.add_axes((0.92, 0.18, 0.012, 0.62))
    colorbar = figure.colorbar(scalar, cax=color_axis)
    colorbar.set_label("$U_3$ from cycle 0 (mm)")
    figure.suptitle(
        "Actual final deformed surfaces after 12 cycles\n"
        "translation removed only; shared physical axes and color scale; no vertical exaggeration",
        fontsize=17,
    )
    figure.subplots_adjust(
        left=0.02, right=0.90, bottom=0.03, top=0.91, wspace=0.0, hspace=0.0
    )
    figure.savefig(output_root / "orth12_deformed_surface_gallery.png", dpi=210)
    plt.close(figure)


def plot_tight_tolerance_surfaces(
    deformations: dict[tuple[float, float], dict[str, np.ndarray]],
    output_root: Path,
) -> None:
    """Give the tol=1e-12 mesh sequence enough space for shape comparison."""

    selected = {
        key: deformation
        for key, deformation in deformations.items()
        if key[1] == 1e-12
    }
    norm = deformation_norm(deformations)
    limits = actual_geometry_limits(selected)
    figure = plt.figure(figsize=(20.0, 5.2))
    for index, resolution in enumerate(RESOLUTIONS, start=1):
        axis = figure.add_subplot(1, len(RESOLUTIONS), index, projection="3d")
        deformation = selected.get((resolution, 1e-12))
        if deformation is None:
            axis.set_axis_off()
            continue
        draw_deformed_surface(axis, deformation, norm, limits)
        axis.set_title(
            f"res {resolution:g}\n$U_3$ span {np.ptp(deformation['u3_mm']):.1f} mm"
        )
    scalar = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
    scalar.set_array([])
    color_axis = figure.add_axes((0.92, 0.19, 0.012, 0.60))
    colorbar = figure.colorbar(scalar, cax=color_axis)
    colorbar.set_label("$U_3$ from cycle 0 (mm)")
    figure.suptitle(
        "Final deformed surfaces at tol $10^{-12}$\n"
        "translation removed only; shared physical axes; no vertical exaggeration",
        fontsize=16,
    )
    figure.subplots_adjust(left=0.01, right=0.90, bottom=0.02, top=0.81, wspace=0.0)
    figure.savefig(output_root / "orth12_tight_tolerance_surfaces.png", dpi=230)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=Path("run/orth12_sensitivity"),
    )
    args = parser.parse_args()
    case_dirs = sorted(path for path in args.output_root.glob("r*_t*") if path.is_dir())
    cases = [case for directory in case_dirs if (case := read_case(directory)) is not None]
    if not cases:
        raise SystemExit(f"No completed cases found under {args.output_root}")
    cases.sort(key=lambda case: (-float(case["resolution"]), -float(case["tolerance"])))
    write_summary(cases, args.output_root)
    plot_sensitivity_grid(cases, args.output_root)
    plot_convergence(cases, args.output_root)
    deformations = load_deformations(cases, args.output_root)
    plot_u3_field_gallery(cases, deformations, args.output_root)
    plot_deformed_surface_gallery(deformations, args.output_root)
    plot_tight_tolerance_surfaces(deformations, args.output_root)
    print(f"Analyzed {len(cases)}/{len(RESOLUTIONS) * len(TOLERANCES)} completed cases")


if __name__ == "__main__":
    main()
