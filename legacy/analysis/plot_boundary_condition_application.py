#!/usr/bin/env python3
"""
Generate a clear, publication-grade diagram showing exactly how the
Centered Clamp Pad (100 mm x 85 mm) and rolling toolpaths are applied
to the 200 mm x 300 mm sheet for Run 6 and Run 9.
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "calibration" / "data" / "raw"
OUT_DIR = ROOT / "calibration" / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_csv_path(path_csv: Path) -> np.ndarray:
    data = []
    with open(path_csv) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if header is None:
                header = parts
                continue
            data.append([float(parts[1]), float(parts[2])])
    return np.array(data)


def main():
    sheet_w = 200.0  # u: [-100, +100]
    sheet_h = 300.0  # v: [-150, +150]
    half_w = sheet_w / 2.0
    half_h = sheet_h / 2.0

    # Load toolpath coordinates
    pts6_raw = load_csv_path(RAW_DIR / "run6_zigzag_path.csv")
    pts9_raw = load_csv_path(RAW_DIR / "run9_zigzag_path.csv")

    pts6_x = pts6_raw[:, 0] - np.mean(pts6_raw[:, 0])
    pts6_v = pts6_raw[:, 1] - 65.0  # v in [-65, +65] mm
    pts6 = np.column_stack([pts6_x, pts6_v])

    pts9_x = pts9_raw[:, 0] - np.mean(pts9_raw[:, 0])
    pts9_v = pts9_raw[:, 1] - 122.5  # v in [-122.5, +65.0] mm
    pts9 = np.column_stack([pts9_x, pts9_v])

    # Clamp rectangle specifications:
    # Centered at (0, 107.5) mm, size: 100 mm x 85 mm -> u in [-50, 50], v in [65, 150]
    clamp_w = 100.0
    clamp_h = 85.0
    clamp_x0 = -clamp_w / 2.0
    clamp_y0 = 65.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 10), dpi=300)
    plt.subplots_adjust(wspace=0.25)

    cases = [
        {
            "ax": axes[0],
            "title": "Run 6: Fixture Clamping + Toolpath Setup",
            "pts": pts6,
            "tool_color": "#1f77b4",
            "tool_name": "Run 6 Toolpath (68 strips, Lv = 130 mm)",
            "tool_span": "v in [-65, +65] mm",
            "bot_margin": "85.0 mm Free Bottom Margin",
        },
        {
            "ax": axes[1],
            "title": "Run 9: Fixture Clamping + Toolpath Setup",
            "pts": pts9,
            "tool_color": "#d62728",
            "tool_name": "Run 9 Toolpath (32 strips, Lv = 187.5 mm)",
            "tool_span": "v in [-122.5, +65] mm",
            "bot_margin": "27.5 mm Free Bottom Margin",
        },
    ]

    for c in cases:
        ax = c["ax"]
        pts = c["pts"]

        # 1. Base Sheet Outline
        rect_sheet = patches.Rectangle(
            (-half_w, -half_h), sheet_w, sheet_h,
            linewidth=2.5, edgecolor="#2c3e50", facecolor="#fdfefe", zorder=1
        )
        ax.add_patch(rect_sheet)

        # 2. Simulate underlying FEM mesh grid (subtle lines)
        for gx in np.linspace(-half_w, half_w, 21):
            ax.axvline(gx, color="#e5e8e8", lw=0.6, zorder=2)
        for gy in np.linspace(-half_h, half_h, 31):
            ax.axhline(gy, color="#e5e8e8", lw=0.6, zorder=2)

        # Coordinate axes
        ax.axhline(0, color="#95a5a6", linestyle="--", lw=0.8, alpha=0.7, zorder=3)
        ax.axvline(0, color="#95a5a6", linestyle="--", lw=0.8, alpha=0.7, zorder=3)

        # 3. The 85 mm Undeformed Fixture Margin Zone (v in [65, 150])
        zone_85 = patches.Rectangle(
            (-half_w, 65.0), sheet_w, 85.0,
            linewidth=1.2, edgecolor="#8e44ad", facecolor="#f4ecf7", alpha=0.5, linestyle="--", zorder=3
        )
        ax.add_patch(zone_85)

        # 4. The Exact Centered Clamp Pad (100 mm x 85 mm)
        clamp_pad = patches.Rectangle(
            (clamp_x0, clamp_y0), clamp_w, clamp_h,
            linewidth=2.2, edgecolor="#9b59b6", facecolor="#af7ac5", alpha=0.65,
            hatch="//", zorder=5, label="Centered Clamp Pad (100 × 85 mm)"
        )
        ax.add_patch(clamp_pad)

        # Mark clamped vertices with purple crosses
        xv = np.linspace(-45, 45, 10)
        yv = np.linspace(70, 145, 8)
        X_clamp, Y_clamp = np.meshgrid(xv, yv)
        ax.scatter(
            X_clamp.ravel(), Y_clamp.ravel(),
            marker="x", color="#4a235a", s=30, lw=1.5, zorder=6,
            label=r"Fixed DOFs ($u_x = u_y = u_z = 0$)"
        )

        # 5. Toolpath strokes
        ax.plot(pts[:, 0], pts[:, 1], color=c["tool_color"], lw=1.1, alpha=0.9, zorder=4, label=c["tool_name"])

        # Toolpath boundary box
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        tool_box = patches.Rectangle(
            (min_x, min_y), max_x - min_x, max_y - min_y,
            linewidth=1.4, edgecolor="#e67e22", facecolor="none", linestyle=":", zorder=7
        )
        ax.add_patch(tool_box)

        # Fixture Datum Line (v = +65.0 mm)
        ax.axhline(65.0, color="#8e44ad", linestyle="-.", lw=1.6, zorder=8)
        ax.text(
            -half_w + 3, 67.5, "v = +65.0 mm (Toolpath Turnaround Datum)",
            color="#8e44ad", fontsize=9, fontweight="bold", zorder=9
        )

        # Dimension annotations
        # Clamp height (85 mm)
        ax.annotate(
            "", xy=(70, 65), xytext=(70, 150),
            arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=2.0), zorder=9
        )
        ax.text(73, 107.5, "85.0 mm\nFixture Zone", color="#8e44ad", fontweight="bold", fontsize=9.5, va="center", zorder=9)

        # Clamp width (100 mm)
        ax.annotate(
            "", xy=(-50, 155), xytext=(50, 155),
            arrowprops=dict(arrowstyle="<->", color="#8e44ad", lw=2.0), zorder=9
        )
        ax.text(0, 160, "100.0 mm Clamp Width", color="#8e44ad", fontweight="bold", fontsize=9.5, ha="center", zorder=9)

        # Lateral free margins
        ax.annotate(
            "", xy=(-half_w, 107.5), xytext=(-50, 107.5),
            arrowprops=dict(arrowstyle="<->", color="#27ae60", lw=1.5), zorder=9
        )
        ax.text(-75, 114, "50 mm\nFree", color="#27ae60", fontweight="bold", fontsize=8.5, ha="center", zorder=9)

        ax.annotate(
            "", xy=(50, 107.5), xytext=(half_w, 107.5),
            arrowprops=dict(arrowstyle="<->", color="#27ae60", lw=1.5), zorder=9
        )
        ax.text(75, 114, "50 mm\nFree", color="#27ae60", fontweight="bold", fontsize=8.5, ha="center", zorder=9)

        # Bottom free margin
        ax.annotate(
            "", xy=(0, -half_h), xytext=(0, min_y),
            arrowprops=dict(arrowstyle="<->", color="#2980b9", lw=1.8), zorder=9
        )
        bot_text_y = (-half_h + min_y) / 2
        ax.text(
            15, bot_text_y, c["bot_margin"],
            color="#2980b9", fontweight="bold", fontsize=9.5, va="center", zorder=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2980b9", alpha=0.85)
        )

        # Footprint summary badge
        ax.text(
            0, (min_y + max_y) / 2,
            f"Active Rolling Footprint:\n{max_x - min_x:.1f} × {max_y - min_y:.1f} mm\n({c['tool_span']})",
            ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdc3c7", alpha=0.92),
            zorder=9
        )

        # Mechanics protocol explanation box at bottom
        protocol_box = (
            "BOUNDARY CONDITION PROTOCOL:\n"
            "• Phase 1 (Growth): 100 × 85 mm pad is rigidly CLAMPED (all DOFs fixed).\n"
            "  Plate deforms under rolling growth while gripped at the top.\n"
            "• Phase 2 (Release): Clamp is completely REMOVED.\n"
            "  Minimal 3-point kinematic gauge applied to resolve springback."
        )
        ax.text(
            -half_w + 5, -half_h + 8, protocol_box,
            fontsize=8.5, color="#1c2833",
            bbox=dict(boxstyle="square,pad=0.4", facecolor="#eaeded", edgecolor="#a6acaf", alpha=0.95),
            zorder=9
        )

        ax.set_xlim(-130, 130)
        ax.set_ylim(-175, 175)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("u / X Material Coordinate (mm)", fontsize=11)
        ax.set_ylabel("v / Y Material Coordinate (mm)", fontsize=11)
        ax.set_title(c["title"], fontsize=12.5, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.legend(loc="upper left", fontsize=8.2, framealpha=0.95)

    plt.tight_layout()
    out_png = OUT_DIR / "boundary_condition_application.png"
    out_pdf = OUT_DIR / "boundary_condition_application.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Generated Boundary Condition setup diagram:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


if __name__ == "__main__":
    main()
