"""Visualize the saved rejected Attempt25 comparison without fitting or rescoring."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from calibration.ewcal.geometry import points, pose_points, read_mesh

CASE = BASE / "run/stage2_bayesian_optimization/evaluations/attempt_0025/case_00"
STEM = BASE / "run/optimization_review/attempt25_saved_surface_comparison"
paths = {ext: STEM.with_suffix("." + ext) for ext in ("png", "pdf", "json")}
caption_path = STEM.with_suffix(".caption.md")
assert os.environ.get("SLURM_JOB_ID"), "Use the approved compute allocation"
assert all(not p.exists() for p in [*paths.values(), caption_path]), "Do not overwrite saved figures"
inputs = [CASE.parent / "result.json", CASE / "prediction_mm.vtp", CASE / "comparison.npz",
          CASE / "bilayer_zigzag_sequence_native_hybrid_state.json",
          BASE / "calibration/data/processed/run6_oriented.npz",
          BASE / "calibration/ewcal/geometry.py"]
hashes = {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
record = json.loads(inputs[0].read_text())
metrics = record["rejected_case"]["metrics"]
assert record["attempt_id"] == 25 and not record["valid"]
assert metrics["stop_reason"] == "iteration_limit" and not metrics["registration_qualified"]
mesh = read_mesh(CASE / "prediction_mm.vtp")
sim = points(mesh)
state = json.loads((CASE / "bilayer_zigzag_sequence_native_hybrid_state.json").read_text())
faces = np.asarray(state["faces"], dtype=int)
assert faces.ndim == 2 and faces.shape[1] == 3 and (faces >= 0).all() and (faces < len(sim)).all()
with np.load(CASE / "comparison.npz", allow_pickle=False) as saved:
    scan = saved["target_mm"].copy()
    distances = saved["distance_mm"].copy()
with np.load(BASE / "calibration/data/processed/run6_oriented.npz", allow_pickle=False) as saved:
    original_scan = saved["points_mm"].copy()
assert scan.shape == original_scan.shape and scan.shape[1] == 3
assert np.isfinite(scan).all() and np.isfinite(sim).all() and np.isfinite(distances).all()
assert distances.shape == (len(scan),) and (distances >= 0).all()
assert np.allclose(scan, pose_points(original_scan, metrics["pose"]), rtol=0, atol=1e-6)
assert abs(float(np.sqrt(np.mean(distances**2))) - metrics["rms_target_to_sim_mm"]) <= 1e-6

# Full geometry/projections and full distance histogram; only 3D scan markers are sampled.
scan_index = np.linspace(0, len(scan) - 1, min(3000, len(scan)), dtype=int)
blue, orange = "#0072B2", "#D55E00"
style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
         "axes.labelsize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
         "figure.facecolor": "white", "axes.facecolor": "white"}
with plt.style.context(style):
    fig = plt.figure(figsize=(10, 7.5), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[3, .9, .8])
    fig.get_layout_engine().set(rect=(0, 0, 1, .92))
    fig.suptitle("Attempt 25 — saved alignment (not converged)", fontsize=13, y=.99)
    legend = fig.legend(handles=[Line2D([], [], color=blue, marker="o", ls="-", label="Simulation"),
                                Line2D([], [], color=orange, marker="x", ls="none", label="Scan")],
                        loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, frameon=False)
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    surface = Poly3DCollection(sim[faces], facecolor=blue, edgecolor="none", alpha=.45)
    surface.set_rasterized(True)
    ax3.add_collection3d(surface)
    ax3.scatter(*scan[scan_index].T, s=2.5, marker="x", c=orange, alpha=.8, rasterized=True)
    combined = np.vstack([sim, scan])
    low, high = combined.min(axis=0), combined.max(axis=0)
    margin = np.maximum((high - low) * .06, 1)
    ax3.set(xlim=(low[0]-margin[0], high[0]+margin[0]), ylim=(low[1]-margin[1], high[1]+margin[1]),
            zlim=(low[2]-margin[2], high[2]+margin[2]), xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    ax3.set_box_aspect(high - low + 2*margin)
    ax3.view_init(elev=24, azim=-62)
    # The physically thin sheet gives a tiny z axis; quantify depth in the projections instead.
    ax3.set_zticks([])
    ax3.set_zlabel("")
    ax3.set_title("(a) Surface overlay", loc="left")
    for row, horizontal, name in ((1, 0, "x–z projection"), (2, 1, "y–z projection")):
        ax = fig.add_subplot(grid[row, :])
        ax.scatter(sim[:, horizontal], sim[:, 2], s=2, c=blue, marker="o", alpha=.3, rasterized=True)
        ax.scatter(scan[:, horizontal], scan[:, 2], s=4, c=orange, marker="x", alpha=.6, rasterized=True)
        ax.set(xlabel=f"{'xy'[horizontal]} (mm)", ylabel="z (mm)",
               xlim=(low[horizontal]-margin[horizontal], high[horizontal]+margin[horizontal]),
               ylim=(low[2]-margin[2], high[2]+margin[2]), aspect="equal")
        ax.set_title(f"({'b' if row == 1 else 'c'}) {name}", loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    ax = fig.add_subplot(grid[0, 1])
    ax.hist(distances, bins=40, range=(0, distances.max()), color=orange, alpha=.85, edgecolor="white")
    ax.axvline(metrics["rms_target_to_sim_mm"], color="black", ls="--", lw=1,
               label=f"Directional RMS {metrics['rms_target_to_sim_mm']:.3f} mm")
    ax.set(xlabel="Scan-to-surface distance (mm)", ylabel="Scan points", xlim=(0, distances.max()))
    ax.set_title("(d) Saved distance distribution", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Fail if figure headings/legends escape the canvas; 3D labels are checked visually too.
    checked = [fig._suptitle, legend, *[a.title for a in fig.axes],
               *[a._left_title for a in fig.axes]]
    for artist in checked:
        if artist.get_visible():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1+1 and box.y1 <= fig.bbox.y1+1, "Text outside canvas"
    fig.savefig(paths["png"], dpi=170)
    fig.savefig(paths["pdf"], dpi=300)
    plt.close(fig)

for p in inputs:
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(BASE))], "Input changed during plotting"
caption = f"""# Attempt25 saved surface comparison

User-selected diagnostic for one Run6 calibration scan, not held-out validation or a representative BO comparison. Blue simulation mesh/circles and orange scan crosses use the existing saved60-iteration registration. (a) All simulation triangles and {len(scan_index)} scan markers selected at evenly spaced indices out of {len(scan)} points; no geometry scaling or new alignment. (b,c) Full point projections, not cross-sections: they can merge distinct surface regions in projection. All spatial views preserve equal physical axis scales. Depth tick labels are omitted from the physically thin3D axis to avoid crowding; both projections quantify z in mm. (d) Full saved scan-to-triangle distance distribution ({len(distances)} correlated points, no uncertainty interval); dashed line is directional RMS, not symmetric RMS.

Saved symmetric RMS **{metrics['rms_symmetric_mm']:.6f}mm**, target→sim **{metrics['rms_target_to_sim_mm']:.6f}mm**, area-weighted sim→target **{metrics['rms_sim_to_target_weighted_mm']:.6f}mm**. Target-distance p95 **{metrics['p95_mm']:.6f}mm**, max **{metrics['max_mm']:.6f}mm**. Candidate remains REJECTED: registration exhausted60 iterations without outer convergence; mechanical solve accepted, pose not near bounds. This is not achievement of the qualified≤3mm target.

The solver output was rigidly registered against its initial mesh using all corresponding vertices before scoring. The processed experimental scan was then rigidly registered to that prediction with rotation≤5deg and translation bounds[5,5,20]mm; saved rotation magnitude {metrics['rotation_angle_deg']:.6f}deg. Neither this plot nor the production rigid transforms flatten the curved surface. The scan surface is treated as the model midsurface; no geometric scale fitting. No edge-based transform has been applied.

Source: `calibration/analysis/plot_attempt25_comparison.py`. Regeneration requires a compute allocation and fresh output paths; provenance/inputs are in `attempt25_saved_surface_comparison.json`.
"""
caption_path.write_text(caption)
paths["json"].write_text(json.dumps({"attempt_id": 25, "qualified": False, "input_sha256": hashes,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "outputs": [str(p.relative_to(BASE)) for p in [paths['png'], paths['pdf'], caption_path]],
    "matplotlib": matplotlib.__version__, "numpy": np.__version__, "python": sys.version,
    "job_id": os.environ['SLURM_JOB_ID'], "hostname": platform.node(),
    "sim_vertices": len(sim), "sim_faces": len(faces), "scan_points": len(scan),
    "scan_indices_3d": scan_index.tolist(), "parameters": record['parameters'],
    "metrics": {k: v for k, v in metrics.items() if k != 'trace'},
    "display": {"dimensions_inches": [10,7.5], "png_dpi": 170, "pdf_raster_dpi": 300,
                "simulation_color": blue, "scan_color": orange, "new_alignment": False}}, indent=2) + "\n")
print(json.dumps({"outputs": {k: str(p) for k, p in paths.items()}, "scan_points": len(scan),
                  "simulation_vertices": len(sim), "saved_rotation_deg": metrics['rotation_angle_deg']}), flush=True)
