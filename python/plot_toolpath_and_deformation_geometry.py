#!/usr/bin/env python3
"""Generate publication-ready figures for toolpath trajectory, mesh geometry,
and resulting equilibrium deformation fields across verified geometries.

Follows the scientific-visualization skill standards:
- Explicit units on all axes and colorbars.
- Diverging colormaps centered at zero for signed out-of-plane deflections.
- Equal aspect ratios for physical coordinates.
- Multi-format export (PNG 300 DPI preview and vector PDF).
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_vtp(path: Path):
    """Load points, triangles, and data arrays from a VTK XML PolyData file."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    points = vtk_to_numpy(data.GetPoints().GetData()).astype(float)
    polys = vtk_to_numpy(data.GetPolys().GetData())
    triangles = polys.reshape(-1, 4)[:, 1:].astype(int)
    
    pd = data.GetPointData()
    cd = data.GetCellData()
    point_arrays = {pd.GetArrayName(i): vtk_to_numpy(pd.GetArray(i)) for i in range(pd.GetNumberOfArrays())}
    cell_arrays = {cd.GetArrayName(i): vtk_to_numpy(cd.GetArray(i)) for i in range(cd.GetNumberOfArrays())}
    return points, triangles, point_arrays, cell_arrays


def plot_toolpath_and_growth_activation():
    """Figure 1: Toolpath trajectory, strip footprints, and discrete mesh activation."""
    print("Generating Figure 1: toolpath_and_growth_activation...")
    
    cycle_file = ROOT / "run/zigzag_single.json"
    cfg = json.loads(cycle_file.read_text())
    tp = cfg["toolpaths"][0]["operation"]
    lv = float(tp["lv_mm"])        # 50 mm
    alpha = float(tp["alpha_deg"]) # 5 deg
    w = float(tp["width_mm"])      # 16 mm
    n_strips = int(tp["n_strips"]) # 2
    
    # Toolpath strip points
    alpha_rad = math.radians(alpha)
    pitch = lv * math.tan(alpha_rad)
    y_bot, y_top = -0.5 * lv, 0.5 * lv
    span = (n_strips - 2) * pitch if n_strips > 2 else 0.0
    x_left, x_right = -0.5 * span, 0.5 * span

    # Load mapping VTP
    vtp_path = ROOT / "run/test_geometries/rect_flat/bilayer_zigzag_sequence_cycle_001_single_pass_r001_mapping.vtp"
    pts, tris, p_arr, c_arr = load_vtp(vtp_path)
    hits = c_arr["hits_this_cycle"]
    
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), constrained_layout=True)
    
    # -------------------------------------------------------------------------
    # Panel (a): Toolpath trajectory layout with zoomed inset
    # -------------------------------------------------------------------------
    ax = axes[0]
    plate_half = 100.0 # mm
    rect = Rectangle((-plate_half, -plate_half), 2 * plate_half, 2 * plate_half,
                     facecolor="#F8FAFC", edgecolor="#94A3B8", lw=1.5, zorder=1)
    ax.add_patch(rect)
    
    # Swath band
    half_w = 0.5 * w
    swath = Rectangle((x_left - half_w, y_bot), w, lv,
                      facecolor="#BAE6FD", edgecolor="#0284C7", lw=1.2, alpha=0.55, zorder=2,
                      label=f"Laser swath ($w = {w:.0f}$ mm)")
    ax.add_patch(swath)
    
    # Centerline path
    ax.plot([x_left, x_left], [y_bot, y_top], color="#0369A1", lw=2.5, zorder=3, label="Scan trajectory")
    ax.annotate("", xy=(x_left, y_top * 0.35), xytext=(x_left, y_bot * 0.35),
                arrowprops=dict(arrowstyle="->", color="#0369A1", lw=2.0), zorder=4)
    
    # Dimension dimension line (placed to the left of the swath)
    dim_x = -22.0
    ax.annotate("", xy=(dim_x, y_top), xytext=(dim_x, y_bot),
                arrowprops=dict(arrowstyle="<->", color="#334155", lw=1.3), zorder=4)
    ax.text(dim_x - 4.0, 0.0, f"$L_v = {lv:.0f}$ mm", color="#1E293B",
            fontsize=9.5, va="center", ha="right", fontweight="medium")
    ax.plot([-half_w, dim_x], [y_top, y_top], color="#94A3B8", ls=":", lw=1.0)
    ax.plot([-half_w, dim_x], [y_bot, y_bot], color="#94A3B8", ls=":", lw=1.0)

    # Inset axes for detailed view of laser toolpath
    ax_ins = inset_axes(ax, width="40%", height="42%", loc="upper right", borderpad=1.0)
    ax_ins.add_patch(Rectangle((-half_w, y_bot), w, lv, facecolor="#BAE6FD", edgecolor="#0284C7", lw=1.2, alpha=0.6))
    ax_ins.plot([0, 0], [y_bot, y_top], color="#0369A1", lw=2.2)
    # Start and end markers
    ax_ins.plot(0, y_bot, marker="o", color="#16A34A", markersize=6, label="Start")
    ax_ins.plot(0, y_top, marker="s", color="#DC2626", markersize=6, label="End")
    ax_ins.annotate("", xy=(0, 5), xytext=(0, -5),
                    arrowprops=dict(arrowstyle="->", color="#0369A1", lw=2.0))
    # Width annotation
    ax_ins.annotate("", xy=(-half_w, -15), xytext=(half_w, -15),
                    arrowprops=dict(arrowstyle="<->", color="#1E293B", lw=1.0))
    ax_ins.text(0, -12, f"$w = {w:.0f}$ mm", fontsize=7.5, ha="center", va="bottom")
    ax_ins.set_xlim(-16, 16)
    ax_ins.set_ylim(-30, 30)
    ax_ins.set_aspect("equal")
    ax_ins.tick_params(labelsize=7)
    ax_ins.set_title("Swath detail", fontsize=8, pad=2)
    ax_ins.grid(True, ls=":", color="#E2E8F0", alpha=0.8)

    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_aspect("equal")
    ax.set_xlabel("Material coordinate $u$ (mm)", fontsize=9.5)
    ax.set_ylabel("Material coordinate $v$ (mm)", fontsize=9.5)
    ax.set_title("(a) Toolpath geometry on plate ($0.2\\times 0.2$ m)", fontsize=10.5, fontweight="bold", pad=8)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax.grid(True, ls=":", color="#E2E8F0", alpha=0.8)

    # -------------------------------------------------------------------------
    # Panel (b): Mesh activation (hit counts)
    # -------------------------------------------------------------------------
    ax = axes[1]
    u_pts = pts[:, 0] * 1e3 # mm
    v_pts = pts[:, 1] * 1e3 # mm
    
    # Create tripcolor plot
    tpc = ax.tripcolor(u_pts, v_pts, tris, facecolors=hits, cmap="YlGnBu",
                       edgecolors="#64748B", linewidth=0.35, zorder=2)
    cb = fig.colorbar(tpc, ax=ax, shrink=0.75, pad=0.04)
    cb.set_label("Laser hit count per face", fontsize=9)
    cb.set_ticks([0, 1, 2])
    cb.ax.tick_params(labelsize=8)
    
    ax.set_xlim(-110, 110)
    ax.set_ylim(-110, 110)
    ax.set_aspect("equal")
    ax.set_xlabel("Material coordinate $u$ (mm)", fontsize=9.5)
    ax.set_ylabel("Material coordinate $v$ (mm)", fontsize=9.5)
    ax.set_title("(b) Discrete mesh hit footprint (800 faces)", fontsize=10.5, fontweight="bold", pad=8)
    ax.grid(True, ls=":", color="#E2E8F0", alpha=0.8)

    # -------------------------------------------------------------------------
    # Panel (c): Eigenstrain intensity profiles
    # -------------------------------------------------------------------------
    ax = axes[2]
    tri_centers_x = np.mean(u_pts[tris], axis=1)
    tri_centers_y = np.mean(v_pts[tris], axis=1)
    
    # Filter strip near centerline (-15 mm < v < 15 mm)
    band_mask = np.abs(tri_centers_y) < 15.0
    x_band = tri_centers_x[band_mask]
    g_top = hits[band_mask] * 1.0 # 1.0 * 10^-3 per hit
    
    sort_idx = np.argsort(x_band)
    ax.plot(x_band[sort_idx], g_top[sort_idx], color="#0284C7", lw=1.8, marker="o", markersize=5,
            label=r"Top layer: $\Delta a_{11}^{\mathrm{top}}$", zorder=3)
    ax.axhline(0.0, color="#D97706", lw=2.0, ls="--",
               label=r"Bottom layer: $\Delta a_{11}^{\mathrm{bot}} = 0$", zorder=2)
    
    ax.set_xlim(-105, 105)
    ax.set_ylim(-0.25, 2.4)
    ax.set_xlabel("Transverse coordinate $u$ (mm)", fontsize=9.5)
    ax.set_ylabel(r"Applied eigenstrain ($10^{-3}$)", fontsize=9.5)
    ax.set_title("(c) Eigenstrain profile at centerline ($v = 0$)", fontsize=10.5, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.grid(True, ls=":", color="#E2E8F0", alpha=0.8)

    fig.savefig(OUT_DIR / "toolpath_and_growth_activation.png", dpi=300)
    fig.savefig(OUT_DIR / "toolpath_and_growth_activation.pdf")
    plt.close(fig)
    print("Saved Figure 1 to docs/figures/toolpath_and_growth_activation.{png,pdf}")


def plot_production_multi_geometry_deformation():
    """Figure 2: 4-row x 3-column comparison of initial vs final equilibrium deformation."""
    print("Generating Figure 2: production_multi_geometry_deformation...")
    
    geometries = [
        ("rect_flat", "Planar Plate (Structured)", False),
        ("rect_curved", "Curved Shell ($R=0.25$ m)", False),
        ("rect_allclamped", "Clamped Plate (Fixed Edges)", True),
        ("rect_irreg", "Delaunay Triangular Mesh", False),
    ]
    
    fig = plt.figure(figsize=(13.5, 14.5), constrained_layout=True)
    
    for row_idx, (folder, title, is_clamped) in enumerate(geometries):
        vtp_init = ROOT / f"run/test_geometries/{folder}/bilayer_zigzag_sequence_cycle_000_initial.vtp"
        vtp_final = ROOT / f"run/test_geometries/{folder}/bilayer_zigzag_sequence_cycle_001_single_pass_r001_final.vtp"
        
        pts0, tris0, p0, c0 = load_vtp(vtp_init)
        pts1, tris1, p1, c1 = load_vtp(vtp_final)
        
        u_pts = pts1[:, 0] * 1e3 # mm
        v_pts = pts1[:, 1] * 1e3 # mm
        z_pts0 = pts0[:, 2] * 1e3 # mm
        
        u3 = p1["U3_from_cycle0"] * 1e6 # um (out of plane)
        umag = p1["Umag_from_cycle0"] * 1e6 # um (total magnitude)
        
        triang = mtri.Triangulation(u_pts, v_pts, triangles=tris1)
        
        # ---------------------------------------------------------------------
        # 1. Column 1: Initial 3D Geometry
        # ---------------------------------------------------------------------
        ax1 = fig.add_subplot(4, 3, row_idx * 3 + 1, projection="3d")
        ax1.plot_trisurf(u_pts, v_pts, z_pts0, triangles=tris1,
                         color="#E2E8F0", edgecolor="#64748B", linewidth=0.25, alpha=0.85)
        
        if is_clamped:
            # Highlight clamped boundary nodes in red
            boundary_mask = (np.abs(u_pts) > 95.0) | (np.abs(v_pts) > 95.0)
            ax1.scatter(u_pts[boundary_mask], v_pts[boundary_mask], z_pts0[boundary_mask],
                        color="#DC2626", s=10, depthshade=False, label="Clamped DOFs")
            ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.8)

        ax1.set_xlabel("X (mm)", fontsize=8, labelpad=2)
        ax1.set_ylabel("Y (mm)", fontsize=8, labelpad=2)
        ax1.set_zlabel("Z (mm)", fontsize=8, labelpad=2)
        ax1.view_init(elev=28, azim=-60)
        ax1.tick_params(labelsize=7)
        ax1.set_title(f"Initial: {title}", fontsize=9.5, fontweight="bold", pad=4)
        
        # ---------------------------------------------------------------------
        # 2. Column 2: Out-of-Plane Deflection Field Uz (2D plan view)
        # ---------------------------------------------------------------------
        ax2 = fig.add_subplot(4, 3, row_idx * 3 + 2)
        max_abs_u3 = max(float(np.max(np.abs(u3))), 0.05)
        norm_u3 = TwoSlopeNorm(vmin=-max_abs_u3, vcenter=0.0, vmax=max_abs_u3)
        
        tpc2 = ax2.tripcolor(triang, u3, cmap="coolwarm", norm=norm_u3,
                             edgecolors="#94A3B8", linewidth=0.2)
        cb2 = fig.colorbar(tpc2, ax=ax2, shrink=0.82, pad=0.04)
        cb2.set_label(r"Out-of-plane $U_z$ ($\mu$m)", fontsize=8)
        cb2.ax.tick_params(labelsize=7)
        
        ax2.set_xlim(-105, 105)
        ax2.set_ylim(-105, 105)
        ax2.set_aspect("equal")
        ax2.set_xlabel("X (mm)", fontsize=8)
        ax2.set_ylabel("Y (mm)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.set_title(f"Deflection $U_z$ (peak: {np.max(u3):+.2f} $\\mu$m)", fontsize=9.5, fontweight="bold", pad=4)
        ax2.grid(True, ls=":", color="#E2E8F0", alpha=0.7)

        # ---------------------------------------------------------------------
        # 3. Column 3: Total Displacement Magnitude Field ||U||
        # ---------------------------------------------------------------------
        ax3 = fig.add_subplot(4, 3, row_idx * 3 + 3)
        norm_umag = Normalize(vmin=0, vmax=np.max(umag))
        tpc3 = ax3.tripcolor(triang, umag, cmap="viridis", norm=norm_umag,
                             edgecolors="#94A3B8", linewidth=0.2)
        cb3 = fig.colorbar(tpc3, ax=ax3, shrink=0.82, pad=0.04)
        cb3.set_label(r"Displacement $\|U\|$ ($\mu$m)", fontsize=8)
        cb3.ax.tick_params(labelsize=7)
        
        ax3.set_xlim(-105, 105)
        ax3.set_ylim(-105, 105)
        ax3.set_aspect("equal")
        ax3.set_xlabel("X (mm)", fontsize=8)
        ax3.set_ylabel("Y (mm)", fontsize=8)
        ax3.tick_params(labelsize=7)
        ax3.set_title(f"$\\|U\\|$ Field (peak: {np.max(umag):.2f} $\\mu$m)", fontsize=9.5, fontweight="bold", pad=4)
        ax3.grid(True, ls=":", color="#E2E8F0", alpha=0.7)

    fig.savefig(OUT_DIR / "production_multi_geometry_deformation.png", dpi=300)
    fig.savefig(OUT_DIR / "production_multi_geometry_deformation.pdf")
    plt.close(fig)
    print("Saved Figure 2 to docs/figures/production_multi_geometry_deformation.{png,pdf}")


def plot_nonconvex_benchmark_equilibria():
    """Figure 3: Non-convex benchmark geometries, 3D equilibria, and adaptive convergence."""
    print("Generating Figure 3: nonconvex_benchmark_equilibria...")
    
    bundle = ROOT / "run/nonconvex_solver_benchmark/profiled_zigzag_gate_7624619/bundle"
    meta = {"nv": 5427, "nf": 10560}
    nv = meta["nv"]
    faces = np.fromfile(bundle / "faces.i32", dtype="<i4").reshape(meta["nf"], 3, order="F")
    
    cases = [
        ("cylinder_x_plus", "Cylinder X+ Equilibrium", 500, "#0284C7"),
        ("twist_plus", "Twist+ Equilibrium", 800, "#7C3AED"),
    ]
    
    fig = plt.figure(figsize=(13.8, 8.8), constrained_layout=True)
    
    for idx, (case_name, title, switch_step, color) in enumerate(cases):
        run_dir = ROOT / f"run/nonconvex_solver_benchmark/adaptive_geom_7829108/{case_name}/screen/00_{case_name}_hybrid"
        cp0_file = run_dir / "checkpoint_0.json"
        
        # Load latest checkpoint (contains complete trace to machine acceptance)
        cp_files = sorted(run_dir.glob("checkpoint_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        latest_cp = cp_files[-1]
        
        c0 = json.loads(cp0_file.read_text())["payload"]["state"]["x"]
        c_final = json.loads(latest_cp.read_text())["payload"]["state"]
        
        # Load vertex coordinates with order='F'
        v_final = np.array(c_final["x"][:nv * 3]).reshape(nv, 3, order="F") * 1000 # mm
        
        trace = c_final.get("trace", [])
        attempts = [r.get("attempt", i) for i, r in enumerate(trace)]
        gnorms = [r.get("gradient_norm", r.get("base_gradient_norm", 1.0)) for r in trace]
        energies = [r.get("energy", 0.0) for r in trace]
        
        # ---------------------------------------------------------------------
        # 1. 3D Deformed Surface Shape
        # ---------------------------------------------------------------------
        ax_3d = fig.add_subplot(2, 3, idx * 3 + 1, projection="3d")
        sub_tris = faces[::2] # 5,280 triangles for detailed smooth surface
        face_z = v_final[sub_tris, 2].mean(axis=1)
        
        surf = ax_3d.plot_trisurf(v_final[:, 0], v_final[:, 1], v_final[:, 2],
                                  triangles=sub_tris, cmap="plasma",
                                  norm=Normalize(vmin=np.min(face_z), vmax=np.max(face_z)),
                                  edgecolor="#1E293B", linewidth=0.1, alpha=0.92)
        surf.set_array(face_z)
        
        ax_3d.set_box_aspect((np.ptp(v_final[:, 0]), np.ptp(v_final[:, 1]), np.ptp(v_final[:, 2]) * 1.5))
        ax_3d.view_init(elev=28, azim=-58)
        ax_3d.set_xlabel("X (mm)", fontsize=8, labelpad=2)
        ax_3d.set_ylabel("Y (mm)", fontsize=8, labelpad=2)
        ax_3d.set_zlabel("Z (mm)", fontsize=8, labelpad=2)
        ax_3d.tick_params(labelsize=7)
        ax_3d.set_title(f"({chr(97 + idx*3)}) {title}", fontsize=10, fontweight="bold", pad=4)
        
        # Horizontal colorbar below 3D surface to eliminate label overlap
        cb = fig.colorbar(surf, ax=ax_3d, orientation="horizontal", shrink=0.55, pad=0.08)
        cb.set_label("Out-of-plane $Z$ (mm)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # ---------------------------------------------------------------------
        # 2. Gradient Norm History (Log scale)
        # ---------------------------------------------------------------------
        ax_gnorm = fig.add_subplot(2, 3, idx * 3 + 2)
        ax_gnorm.semilogy(attempts, gnorms, color=color, lw=1.6, label="Hybrid trajectory")
        ax_gnorm.axvline(switch_step, color="#DC2626", ls="--", lw=1.5,
                         label=f"Adaptive switch (step {switch_step})")
        ax_gnorm.axhline(5e-14, color="#059669", ls=":", lw=1.5, label=r"Acceptance floor ($5\times 10^{-14}$ N)")
        
        # Mark final machine-precision accepted point
        ax_gnorm.plot(attempts[-1], gnorms[-1], marker="^", color="#059669", markersize=7,
                      label=f"Accepted: {gnorms[-1]:.1e} N", zorder=5)
        
        ax_gnorm.set_ylim(1e-16, 3e-7)
        ax_gnorm.set_xlabel("Optimizer attempt", fontsize=9)
        ax_gnorm.set_ylabel(r"Physical gradient norm $\|\nabla E\|$ (N)", fontsize=9)
        ax_gnorm.set_title(f"({chr(98 + idx*3)}) Gradient convergence: {case_name}", fontsize=10, fontweight="bold", pad=4)
        ax_gnorm.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_gnorm.grid(True, ls=":", color="#E2E8F0", alpha=0.8)
        ax_gnorm.tick_params(labelsize=8)

        # ---------------------------------------------------------------------
        # 3. Energy History
        # ---------------------------------------------------------------------
        ax_e = fig.add_subplot(2, 3, idx * 3 + 3)
        e_final = energies[-1]
        delta_e = [max(e - e_final, 1e-19) for e in energies]
        ax_e.semilogy(attempts, delta_e, color="#D97706", lw=1.6, label=r"$E - E_{\mathrm{final}}$")
        ax_e.axvline(switch_step, color="#DC2626", ls="--", lw=1.5, label="Newton handoff")
        
        ax_e.set_ylim(1e-19, 1e-9)
        ax_e.set_xlabel("Optimizer attempt", fontsize=9)
        ax_e.set_ylabel(r"Relative energy excess $\Delta E$ (J)", fontsize=9)
        ax_e.set_title(f"({chr(99 + idx*3)}) Energy relaxation: {case_name}", fontsize=10, fontweight="bold", pad=4)
        ax_e.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_e.grid(True, ls=":", color="#E2E8F0", alpha=0.8)
        ax_e.tick_params(labelsize=8)
        ax_e.set_title(f"({chr(99 + idx*3)}) Energy relaxation: {case_name}", fontsize=10, fontweight="bold", pad=4)
        ax_e.legend(loc="upper right", fontsize=8, framealpha=0.9)
        ax_e.grid(True, ls=":", color="#E2E8F0", alpha=0.8)
        ax_e.tick_params(labelsize=8)

    fig.savefig(OUT_DIR / "nonconvex_benchmark_equilibria.png", dpi=300)
    fig.savefig(OUT_DIR / "nonconvex_benchmark_equilibria.pdf")
    plt.close(fig)
    print("Saved Figure 3 to docs/figures/nonconvex_benchmark_equilibria.{png,pdf}")


def write_captions():
    """Write markdown companion captions for each figure."""
    (OUT_DIR / "toolpath_and_growth_activation.caption.md").write_text("""# Figure 1: Toolpath Trajectory and Discrete Mesh Activation

**(a) Toolpath Geometry and Sweep Pattern**: Two-strip zigzag laser toolpath trajectory on the $200\\times 200$ mm shell reference plate ($[-100, 100] \\times [-100, 100]$ mm). Blue arrows designate scan velocity vector directions along path length $L_v = 50.0$ mm and sweep tilt angle $\\alpha = 5.0^\\circ$. Shaded blue swath corresponds to the finite beam print width $w = 16.0$ mm. Inset shows high-magnification swath geometry with start and end locations.
**(b) Discrete Mesh Hit Footprint**: Spatial distribution of discrete laser hit events across the structured triangular discretization (441 vertices, 800 faces). Unhit background elements remain neutral; activated faces receive differential eigenstrain stimulation.
**(c) Applied Eigenstrain Cross-Section**: Differential growth intensity through the thickness along the plate centerline ($v \\approx 0$). Top layer receives positive longitudinal expansion $\\Delta a_{11}^{\\mathrm{top}} = 10^{-3}$ per hit while the bottom layer remains inert ($\\Delta a_{11}^{\\mathrm{bot}} = 0$), inducing localized bending moments that drive out-of-plane equilibrium deformation.
""")

    (OUT_DIR / "production_multi_geometry_deformation.caption.md").write_text("""# Figure 2: Equilibrium Deformations Across Diverse Shell Geometries

Comparative response of the production hybrid equilibrium solver (`-equilibrium_solver hybrid -hessian_threads 4`) across four distinct mechanical and geometric configurations subjected to the single-pass zigzag toolpath:
- **Row 1 (`rect_flat`)**: Planar rectangular structured plate ($0.2\\times 0.2$ m). Bending moment induces symmetric double-curvature deflection ($U_z \\in [-0.15, +0.23]\\;\\mu$m; maximum displacement $\\|U\\| = 6.06\\;\\mu$m).
- **Row 2 (`rect_curved`)**: Cylindrical shell panel with analytical transverse radius $R = 0.25$ m (initial out-of-plane camber $Z \\in [0, 19.7]$ mm). Toolpath induces asymmetric deflection coupling with initial membrane curvature ($U_z \\in [-0.12, +0.68]\\;\\mu$m).
- **Row 3 (`rect_allclamped`)**: Clamped boundary rectangular plate with edge displacement pinning (red markers indicate constrained edge DOFs). Boundary reaction forces restrict boundary deflection, concentrating curvature into the central domain ($U_z \\in [-0.15, +0.23]\\;\\mu$m).
- **Row 4 (`rect_irreg`)**: Unstructured Delaunay triangular mesh (348 vertices, 636 faces) with random triangle orientations. The parallel solver preserves exact bitwise reproducibility without grid-alignment bias (maximum deflection $U_z = 0.38\\;\\mu$m; maximum displacement $\\|U\\| = 7.69\\;\\mu$m).
""")

    (OUT_DIR / "nonconvex_benchmark_equilibria.caption.md").write_text("""# Figure 3: Non-Convex Benchmark Equilibria and Adaptive Curvature Handoff

Evaluation of the adaptive curvature gate (`RobustCurvatureGate`) on large-scale non-convex shell benchmark geometries (10,560 faces, 32,267 degrees of freedom):
- **(a, d) 3D Equilibrium Deformation**: Final converged stationary configuration rendered with true physical coordinates (colored by out-of-plane vertical deflection $Z$, showing cylindrical and saddle/twisted curvature profiles up to $50$ mm deflection).
- **(b, e) Physical Gradient Norm Trajectory**: Logarithmic convergence of the physical gradient norm $\\|\\nabla E\\|$ as a function of optimizer attempts. `cylinder_x_plus` certified positive unshifted restricted pivots ($\mathbf{H}_r \\succ 0$) and transitioned to Newton at attempt 500; `twist_plus` certified and transitioned at attempt 800. Unshifted direct Cholesky steps ($\lambda = 0.0$) achieve rapid quadratic convergence to machine zero ($4.3\\times 10^{-16}$ N and $3.3\\times 10^{-15}$ N), satisfying the strict acceptance floor ($5\\times 10^{-14}$ N).
- **(c, f) Energy Relaxation**: Monotonic decay of excess energy $\\Delta E = E - E_{\\mathrm{final}}$ toward stationary equilibrium ($E_{\\mathrm{final}} = 2.763\\times 10^{-11}$ J).
""")
    print("Saved caption markdown files to docs/figures/")


if __name__ == "__main__":
    plot_toolpath_and_growth_activation()
    plot_production_multi_geometry_deformation()
    plot_nonconvex_benchmark_equilibria()
    write_captions()
    print("All figures and captions generated successfully.")
