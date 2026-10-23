"""Audit sequence target metrics in material coordinates from a final VTP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def cell_array(data, name: str) -> np.ndarray:
    array = data.GetCellData().GetArray(name)
    if array is None:
        raise RuntimeError(f"Missing cell array {name!r}")
    return vtk_to_numpy(array).astype(float)


def weighted_stats(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    selected = values[mask]
    selected_weights = weights[mask]
    if not len(selected):
        return {"count": 0}
    order = np.argsort(selected)
    sorted_values = selected[order]
    cumulative = np.cumsum(selected_weights[order])
    cumulative /= cumulative[-1]

    def quantile(probability: float) -> float:
        return float(sorted_values[np.searchsorted(cumulative, probability, side="left")])

    return {
        "count": int(len(selected)),
        "area_fraction": float(np.sum(selected_weights) / np.sum(weights)),
        "mean": float(np.average(selected, weights=selected_weights)),
        "median": quantile(0.5),
        "p90": quantile(0.9),
        "p99": quantile(0.99),
        "max": float(np.max(selected)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("final_vtp", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--compare-vtp",
        type=Path,
        help="face-aligned final VTP whose target tensors are compared to final_vtp",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.final_vtp.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(args.final_vtp))
    reader.Update()
    data = reader.GetOutput()
    point_data = data.GetPointData()
    material_u_array = point_data.GetArray("material_u")
    material_v_array = point_data.GetArray("material_v")
    if material_u_array is None or material_v_array is None:
        raise RuntimeError("Missing material_u/material_v point arrays")
    material = np.column_stack(
        (
            vtk_to_numpy(material_u_array).astype(float),
            vtk_to_numpy(material_v_array).astype(float),
        )
    )
    triangles = (
        vtk_to_numpy(data.GetPolys().GetData())
        .reshape(-1, 4)[:, 1:]
        .astype(int)
    )
    n_faces = len(triangles)
    material_metrics = {}
    edge_bases = np.empty((n_faces, 2, 2), dtype=float)
    material_area = np.empty(n_faces, dtype=float)
    for face, (i0, i1, i2) in enumerate(triangles):
        Dm = np.column_stack((material[i2] - material[i1], material[i0] - material[i2]))
        edge_bases[face] = Dm
        material_area[face] = 0.5 * abs(np.linalg.det(Dm))

    for layer in ("top", "bot"):
        a11 = cell_array(data, f"abar_{layer}_11")
        a12 = cell_array(data, f"abar_{layer}_12")
        a22 = cell_array(data, f"abar_{layer}_22")
        metrics = np.empty((n_faces, 2, 2), dtype=float)
        for face in range(n_faces):
            edge_metric = np.array([[a11[face], a12[face]], [a12[face], a22[face]]])
            inverse = np.linalg.inv(edge_bases[face])
            metrics[face] = inverse.T @ edge_metric @ inverse
        material_metrics[layer] = metrics

    top = material_metrics["top"]
    bottom = material_metrics["bot"]
    top_eigenvalues = np.linalg.eigvalsh(top)
    top_mean = np.mean(top_eigenvalues, axis=1)
    top_isotropy_error = (
        np.abs(top_eigenvalues[:, 1] - top_eigenvalues[:, 0])
        / np.maximum(np.abs(top_mean), 1e-15)
    )
    top_offdiag_error = np.abs(top[:, 0, 1]) / np.maximum(np.abs(top_mean), 1e-15)
    bottom_identity_error = np.linalg.norm(bottom - np.eye(2), axis=(1, 2))

    hits_cycle2 = cell_array(data, "hits_this_cycle")
    hits_total = cell_array(data, "total_hit_count")
    hits_cycle1 = hits_total - hits_cycle2
    treated = hits_total > 0.0
    paired = (hits_cycle1 > 0.0) & (hits_cycle2 > 0.0)
    paired_equal = paired & (hits_cycle1 == hits_cycle2)

    weighted_mean_top = np.average(top, axis=0, weights=material_area * treated)
    weighted_mean_eigenvalues = np.linalg.eigvalsh(weighted_mean_top)
    weighted_mean_isotropy_error = float(
        abs(weighted_mean_eigenvalues[1] - weighted_mean_eigenvalues[0])
        / np.mean(weighted_mean_eigenvalues)
    )

    comparison = None
    if args.compare_vtp is not None:
        comparison_reader = vtk.vtkXMLPolyDataReader()
        comparison_reader.SetFileName(str(args.compare_vtp))
        comparison_reader.Update()
        comparison_data = comparison_reader.GetOutput()
        comparison_triangles = (
            vtk_to_numpy(comparison_data.GetPolys().GetData())
            .reshape(-1, 4)[:, 1:]
            .astype(int)
        )
        comparison_material = np.column_stack(
            (
                vtk_to_numpy(
                    comparison_data.GetPointData().GetArray("material_u")
                ).astype(float),
                vtk_to_numpy(
                    comparison_data.GetPointData().GetArray("material_v")
                ).astype(float),
            )
        )
        if not np.array_equal(triangles, comparison_triangles):
            raise RuntimeError("Comparison VTP does not have face-aligned topology")
        if not np.allclose(material, comparison_material, rtol=0.0, atol=1e-14):
            raise RuntimeError("Comparison VTP does not have matching material coordinates")
        target_arrays = (
            "abar_top_11",
            "abar_top_12",
            "abar_top_22",
            "abar_bot_11",
            "abar_bot_12",
            "abar_bot_22",
        )
        comparison = {
            "source": str(args.compare_vtp),
            "max_abs_target_array_difference": {
                name: float(
                    np.max(
                        np.abs(
                            cell_array(data, name)
                            - cell_array(comparison_data, name)
                        )
                    )
                )
                for name in target_arrays
            },
        }
    report = {
        "source": str(args.final_vtp),
        "faces": n_faces,
        "cycle1_hit_events": int(np.sum(hits_cycle1)),
        "cycle2_hit_events": int(np.sum(hits_cycle2)),
        "treated_faces": int(np.sum(treated)),
        "paired_faces": int(np.sum(paired)),
        "equal_multiplicity_paired_faces": int(np.sum(paired_equal)),
        "weighted_mean_top_metric": weighted_mean_top.tolist(),
        "weighted_mean_top_metric_eigenvalues": weighted_mean_eigenvalues.tolist(),
        "weighted_mean_top_isotropy_error": weighted_mean_isotropy_error,
        "bottom_identity_error_max": float(np.max(bottom_identity_error)),
        "top_isotropy_error_treated": weighted_stats(
            top_isotropy_error, material_area, treated
        ),
        "top_isotropy_error_paired": weighted_stats(
            top_isotropy_error, material_area, paired
        ),
        "top_isotropy_error_equal_multiplicity": weighted_stats(
            top_isotropy_error, material_area, paired_equal
        ),
        "top_offdiag_error_paired": weighted_stats(
            top_offdiag_error, material_area, paired
        ),
        "hit_imbalance_paired": weighted_stats(
            np.abs(hits_cycle1 - hits_cycle2), material_area, paired
        ),
    }
    if comparison is not None:
        report["comparison"] = comparison
    stem = f"{args.final_vtp.parent.name}_{args.final_vtp.stem}"
    report_path = output_dir / f"{stem}_target_metric_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    u_mm = 1000.0 * material[:, 0]
    v_mm = 1000.0 * material[:, 1]
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 10.5), constrained_layout=True)
    panels = (
        (top_isotropy_error, "top target-metric isotropy error", "magma", None),
        (top_offdiag_error, "normalized |top metric off-diagonal|", "viridis", None),
        (hits_cycle1 - hits_cycle2, "cycle-1 minus cycle-2 hit count", "coolwarm", 0.0),
        (top_mean - 1.0, "mean top target-metric increment", "plasma", None),
    )
    for axis, (values, title, cmap_name, center) in zip(axes.flat, panels):
        norm = None
        if center is not None and np.min(values) < center < np.max(values):
            norm = colors.TwoSlopeNorm(
                vmin=float(np.min(values)), vcenter=center, vmax=float(np.max(values))
            )
        image = axis.tripcolor(
            u_mm,
            v_mm,
            triangles,
            facecolors=values,
            shading="flat",
            cmap=cmap_name,
            norm=norm,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("material u (mm)")
        axis.set_ylabel("material v (mm)")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle(
        "Sequence target metric in material coordinates\n"
        f"mean isotropy error={weighted_mean_isotropy_error:.3e}; "
        f"cycle hits={int(np.sum(hits_cycle1))}/{int(np.sum(hits_cycle2))}",
        fontsize=15,
    )
    figure.savefig(output_dir / f"{stem}_target_metric_audit.png", dpi=210)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
