"""Compare corrected one-shot and continuation loading trajectories."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors as mpl_colors
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from analyze_hlbfgs_resolution_sweep import (
    SUMMARY_NAME,
    _interpolate,
    _read_case,
    _remove_plane,
)


DEFAULT_CASES = (
    ("one_shot", "pt_1step"),
    ("uniform_10", "pt_uniform10"),
    ("stripwise_30", "pt_traj30"),
)


def _summary_rows(directory: Path) -> list[dict[str, str]]:
    with (directory / SUMMARY_NAME).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No trajectory rows in {directory / SUMMARY_NAME}")
    return rows


def _wall_seconds(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"(?:WALL |wall_seconds=)([0-9.]+)(?: s)?", text)
    return float(matches[-1]) if matches else float("nan")


def analyze(
    root: Path,
) -> tuple[
    list[dict[str, float | int | str]],
    dict[str, list[float]],
    dict[str, dict[str, np.ndarray]],
]:
    loaded = [
        (label, _read_case(root / directory), _summary_rows(root / directory))
        for label, directory in DEFAULT_CASES
    ]
    reference = loaded[0][1]
    reference_field = _remove_plane(reference["u"], reference["v"], reference["u3"])
    reference_rms = float(np.sqrt(np.mean(reference_field**2)))

    results: list[dict[str, float | int | str]] = []
    histories: dict[str, list[float]] = {}
    curvatures: dict[str, dict[str, np.ndarray]] = {}
    for label, case, summary in loaded:
        interpolated = _interpolate(case, reference["u"], reference["v"])
        field = _remove_plane(reference["u"], reference["v"], interpolated)
        direct = float(np.sqrt(np.mean((field - reference_field) ** 2)))
        reflected = float(np.sqrt(np.mean((-field - reference_field) ** 2)))
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(case["directory"] / summary[-1]["final_file"]))
        reader.Update()
        mean_curvature = vtk_to_numpy(
            reader.GetOutput().GetCellData().GetArray("mean")
        ).astype(float)
        triangles = case["triangles"]
        points = case["points"]
        cross = np.cross(
            points[triangles[:, 1]] - points[triangles[:, 0]],
            points[triangles[:, 2]] - points[triangles[:, 0]],
        )
        area = 0.5 * np.linalg.norm(cross, axis=1)
        total_area = float(np.sum(area))
        area_weighted_mean = float(np.sum(area * mean_curvature) / total_area)
        area_weighted_rms = float(
            np.sqrt(np.sum(area * mean_curvature**2) / total_area)
        )
        results.append(
            {
                "scheme": label,
                "cycles": len(summary),
                "resolution_m": case["resolution_m"],
                "faces": case["faces"],
                "energy": case["energy"],
                "energy_relative_difference": abs(
                    case["energy"] / reference["energy"] - 1.0
                ),
                "u3_span_mm": 1000.0 * case["u3_span_m"],
                "u3_span_relative_difference": abs(
                    case["u3_span_m"] / reference["u3_span_m"] - 1.0
                ),
                "abs_mean_curvature_integral": case["abs_mean_curvature_integral"],
                "signed_mean_curvature_integral": case[
                    "signed_mean_curvature_integral"
                ],
                "area_weighted_mean_curvature_per_m": area_weighted_mean,
                "area_weighted_rms_curvature_per_m": area_weighted_rms,
                "minimum_mean_curvature_per_m": float(np.min(mean_curvature)),
                "maximum_mean_curvature_per_m": float(np.max(mean_curvature)),
                "curvature_relative_difference": abs(
                    case["abs_mean_curvature_integral"]
                    / reference["abs_mean_curvature_integral"]
                    - 1.0
                ),
                "field_nrmse": min(direct, reflected) / max(reference_rms, 1e-15),
                "wall_seconds": _wall_seconds(case["directory"] / "run.log"),
            }
        )
        curvatures[label] = {
            "u": case["u"],
            "v": case["v"],
            "points": points,
            "triangles": triangles,
            "mean": mean_curvature,
        }
        histories[f"{label}_progress"] = [
            (index + 1) / len(summary) for index in range(len(summary))
        ]
        histories[f"{label}_energy"] = [float(row["total_energy"]) for row in summary]
        histories[f"{label}_span_mm"] = [
            1000.0 * (float(row["max_U3"]) - float(row["min_U3"]))
            for row in summary
        ]
    return results, histories, curvatures


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, float | int | str]],
    histories: dict[str, list[float]],
    path: Path,
) -> None:
    colors = {
        "one_shot": "#4c78a8",
        "uniform_10": "#f58518",
        "stripwise_30": "#54a24b",
    }
    labels = {
        "one_shot": "one shot",
        "uniform_10": "uniform (10)",
        "stripwise_30": "stripwise (30)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for scheme in colors:
        progress = histories[f"{scheme}_progress"]
        axes[0].plot(
            progress,
            np.asarray(histories[f"{scheme}_energy"]) * 1e11,
            "-o",
            ms=3,
            color=colors[scheme],
            label=labels[scheme],
        )
        axes[1].plot(
            progress,
            histories[f"{scheme}_span_mm"],
            "-o",
            ms=3,
            color=colors[scheme],
            label=labels[scheme],
        )

    schemes = [str(row["scheme"]) for row in rows]
    x = np.arange(len(schemes))
    width = 0.25
    endpoint_metrics = (
        ("field NRMSE", "field_nrmse"),
        ("span difference", "u3_span_relative_difference"),
        ("energy difference", "energy_relative_difference"),
    )
    for offset, (label, key) in enumerate(endpoint_metrics):
        axes[2].bar(
            x + (offset - 1) * width,
            [100.0 * float(row[key]) for row in rows],
            width,
            label=label,
        )
    axes[2].set_xticks(x, [labels[scheme] for scheme in schemes], rotation=15)
    axes[2].set_ylabel("difference from one shot (%)")
    axes[2].legend(fontsize=8)

    axes[0].set_ylabel(r"total energy ($10^{-11}$ J)")
    axes[1].set_ylabel(r"$u_3$ span (mm)")
    for axis in axes[:2]:
        axis.set_xlabel("fraction of final applied growth")
        axis.set_xlim(0.0, 1.02)
        axis.legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Loading-trajectory ablation at HLBFGS tolerance $10^{-12}$")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_curvature(
    rows: list[dict[str, float | int | str]],
    curvatures: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    labels = {
        "one_shot": "one shot",
        "uniform_10": "uniform continuation (10)",
        "stripwise_30": "stripwise trajectory (30)",
    }
    all_curvature = np.concatenate(
        [curvatures[str(row["scheme"])]["mean"] for row in rows]
    )
    norm = mpl_colors.TwoSlopeNorm(
        vmin=float(np.min(all_curvature)),
        vcenter=0.0,
        vmax=float(np.max(all_curvature)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    image = None
    for axis, row in zip(axes, rows):
        scheme = str(row["scheme"])
        data = curvatures[scheme]
        image = axis.tripcolor(
            1000.0 * data["u"],
            1000.0 * data["v"],
            data["triangles"],
            facecolors=data["mean"],
            cmap="coolwarm",
            norm=norm,
            shading="flat",
            rasterized=True,
        )
        axis.set_title(
            f"{labels[scheme]}\n"
            f"area mean {float(row['area_weighted_mean_curvature_per_m']):+.3f} "
            r"m$^{-1}$"
        )
        axis.set_xlabel("material $u$ (mm)")
        axis.set_aspect("equal")
    axes[0].set_ylabel("material $v$ (mm)")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.86, pad=0.02)
        colorbar.set_label(r"signed mean curvature $H$ (m$^{-1}$), full range")
    fig.suptitle("Final mean-curvature field at HLBFGS tolerance $10^{-12}$")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_curvature_3d(
    rows: list[dict[str, float | int | str]],
    curvatures: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    labels = {
        "one_shot": "one shot",
        "uniform_10": "uniform continuation (10)",
        "stripwise_30": "stripwise trajectory (30)",
    }
    all_curvature = np.concatenate(
        [curvatures[str(row["scheme"])]["mean"] for row in rows]
    )
    norm = mpl_colors.TwoSlopeNorm(
        vmin=float(np.min(all_curvature)),
        vcenter=0.0,
        vmax=float(np.max(all_curvature)),
    )
    all_points_mm = np.concatenate(
        [1000.0 * curvatures[str(row["scheme"])]["points"] for row in rows]
    )
    limits = [
        (float(np.min(all_points_mm[:, axis])), float(np.max(all_points_mm[:, axis])))
        for axis in range(3)
    ]
    spans = [upper - lower for lower, upper in limits]

    fig = plt.figure(figsize=(16.2, 5.4))
    axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    cmap = plt.get_cmap("coolwarm")
    for axis, row in zip(axes, rows):
        scheme = str(row["scheme"])
        data = curvatures[scheme]
        points_mm = 1000.0 * data["points"]
        surface = axis.plot_trisurf(
            points_mm[:, 0],
            points_mm[:, 1],
            points_mm[:, 2],
            triangles=data["triangles"],
            linewidth=0.02,
            antialiased=True,
            shade=False,
        )
        surface.set_facecolors(cmap(norm(data["mean"])))
        axis.set_title(
            f"{labels[scheme]}\n"
            rf"$u_3$ span {float(row['u3_span_mm']):.2f} mm",
            pad=8,
        )
        axis.set_xlim(*limits[0])
        axis.set_ylim(*limits[1])
        axis.set_zlim(*limits[2])
        axis.set_box_aspect((spans[0], spans[1], 3.0 * spans[2]))
        axis.view_init(elev=27.0, azim=-58.0)
        axis.set_xlabel("x (mm)", labelpad=4)
        axis.set_ylabel("y (mm)", labelpad=4)
        axis.set_zlabel("z (mm)", labelpad=4)
        axis.tick_params(labelsize=8, pad=1)
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    color_axis = fig.add_axes((0.30, 0.055, 0.40, 0.025))
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label(r"signed mean curvature $H$ (m$^{-1}$), full range")
    fig.suptitle(
        "Deformed final surfaces colored by mean curvature "
        "(3x vertical exaggeration)",
        fontsize=16,
        y=0.98,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.15, top=0.86, wspace=0.02)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path("run/zigzag_production"),
    )
    args = parser.parse_args()
    rows, histories, curvatures = analyze(args.root)
    csv_path = args.root / "trajectory_ablation_summary.csv"
    figure_path = args.root / "trajectory_ablation_history.png"
    curvature_path = args.root / "trajectory_ablation_curvature.png"
    curvature_3d_path = args.root / "trajectory_ablation_curvature_3d.png"
    write_csv(rows, csv_path)
    plot(rows, histories, figure_path)
    plot_curvature(rows, curvatures, curvature_path)
    plot_curvature_3d(rows, curvatures, curvature_3d_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {figure_path}")
    print(f"Wrote {curvature_path}")
    print(f"Wrote {curvature_3d_path}")


if __name__ == "__main__":
    main()
