#!/usr/bin/env python3
"""
Plot the exact toolpaths for Run 6 and Run 9 on the 200 mm x 300 mm sheet,
accounting for the mandatory 85.0 mm fixture clearance at the bottom edge.
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
            # parts[1] is x_mm, parts[2] is y_mm
            data.append([float(parts[1]), float(parts[2])])
    return np.array(data)


def main():
    # Sheet dimensions: 200 mm wide (u / X) x 300 mm long (v / Y)
    sheet_w = 200.0
    sheet_h = 300.0
    half_w = sheet_w / 2.0  # 100.0 mm
    half_h = sheet_h / 2.0  # 150.0 mm

    # Load raw toolpaths (in raw CSV, y runs from 0 to Lv)
    pts6_raw = load_csv_path(RAW_DIR / "run6_zigzag_path.csv")
    pts9_raw = load_csv_path(RAW_DIR / "run9_zigzag_path.csv")

    # Center in X (u)
    pts6_x = pts6_raw[:, 0] - np.mean(pts6_raw[:, 0])
    pts9_x = pts9_raw[:, 0] - np.mean(pts9_raw[:, 0])

    # MANDATORY RULE: Bottom margin must be exactly 85.0 mm from the bottom edge (v = -150.0 mm).
    # Therefore, the bottom turnaround of the toolpath must sit at v = -150.0 + 85.0 = -65.0 mm.
    v_bottom_target = -half_h + 85.0  # -65.0 mm

    # Run 6: Lv = 130 mm -> v runs from -65.0 to +65.0 mm (Top margin: 150 - 65 = 85.0 mm)
    pts6_v = v_bottom_target + pts6_raw[:, 1]
    pts6_corrected = np.column_stack([pts6_x, pts6_v])

    # Run 9 Corrected: Lv = 187.5 mm -> v runs from -65.0 to +122.5 mm (Top margin: 150 - 122.5 = 27.5 mm)
    pts9_v_corrected = v_bottom_target + pts9_raw[:, 1]
    pts9_corrected = np.column_stack([pts9_x, pts9_v_corrected])

    # Run 9 Erroneous / Old Config (shift = 0, centered symmetrically):
    # v runs from -93.75 to +93.75 mm (Bottom margin: 56.25 mm != 85.0 mm)
    pts9_v_old = pts9_raw[:, 1] - (np.max(pts9_raw[:, 1]) / 2.0)
    pts9_old = np.column_stack([pts9_x, pts9_v_old])

    # Create 3-panel figure:
    # Panel 1: Run 6 (85 mm bottom margin)
    # Panel 2: Run 9 Corrected (85 mm bottom margin, 27.5 mm top margin)
    # Panel 3: Run 9 Comparison (Corrected 85 mm bottom margin vs Erroneous 56.2 mm bottom margin)
    fig, axes = plt.subplots(1, 3, figsize=(18, 9), dpi=300)

    sheet_color = "#f8f9fa"
    sheet_edge = "#2c3e50"

    panels = [
        {
            "ax": axes[0],
            "title": "Run 6 (Lv = 130 mm)\nMandatory 85 mm Bottom Margin",
            "pts": pts6_corrected,
            "color": "#1f77b4",
            "v_bot": -65.0,
            "v_top": 65.0,
            "bot_margin": 85.0,
            "top_margin": 85.0,
            "left_margin": half_w - np.max(np.abs(pts6_x)),
            "note": "Lv = 130 mm, 68 strips\nSymmetric by coincidence:\n85 mm bottom + 85 mm top",
        },
        {
            "ax": axes[1],
            "title": "Run 9 CORRECTED (Lv = 187.5 mm)\nMandatory 85 mm Bottom Margin",
            "pts": pts9_corrected,
            "color": "#d62728",
            "v_bot": -65.0,
            "v_top": 122.5,
            "bot_margin": 85.0,
            "top_margin": 27.5,
            "left_margin": half_w - np.max(np.abs(pts9_x)),
            "note": "Lv = 187.5 mm, 32 strips\nFixed 85.0 mm bottom margin\nTop margin = 27.5 mm\nToolpath center shift: +28.75 mm",
        },
        {
            "ax": axes[2],
            "title": "Run 9 Alignment Discrepancy\nCorrect (85 mm) vs Config (56.2 mm)",
            "pts_correct": pts9_corrected,
            "pts_old": pts9_old,
            "note": "Orange dashed: Old run9.json (shift=[0,0])\n  -> Erroneous 56.2 mm bottom margin\nRed solid: Correct physical alignment\n  -> True 85.0 mm bottom margin\nUpward offset required: +28.75 mm",
        },
    ]

    # Plot Panels 1 and 2
    for p in panels[:2]:
        ax = p["ax"]
        pts = p["pts"]

        # Sheet outline
        rect = patches.Rectangle(
            (-half_w, -half_h), sheet_w, sheet_h,
            linewidth=2.0, edgecolor=sheet_edge, facecolor=sheet_color, zorder=1
        )
        ax.add_patch(rect)

        # Centerline axes
        ax.axhline(0, color="#95a5a6", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)
        ax.axvline(0, color="#95a5a6", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

        # Fixture baseline (v = -65 mm)
        ax.axhline(-65.0, color="#8e44ad", linestyle="-.", linewidth=1.2, alpha=0.85, zorder=3)

        # Toolpath
        ax.plot(pts[:, 0], pts[:, 1], color=p["color"], linewidth=0.9, alpha=0.9, zorder=4)

        # Bounding box
        min_x, max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
        min_y, max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
        box = patches.Rectangle(
            (min_x, min_y), max_x - min_x, max_y - min_y,
            linewidth=1.2, edgecolor="#e67e22", facecolor="none", linestyle=":", zorder=5
        )
        ax.add_patch(box)

        # Bottom margin arrow (MANDATORY 85 mm)
        ax.annotate(
            "", xy=(0, -half_h), xytext=(0, min_y),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2.2), zorder=6
        )
        ax.text(
            6, (-half_h + min_y) / 2,
            f"MANDATORY\nBOTTOM MARGIN:\n{p['bot_margin']:.1f} mm",
            color="#c0392b", fontweight="bold", fontsize=10, va="center", zorder=6
        )

        # Top margin arrow
        ax.annotate(
            "", xy=(0, half_h), xytext=(0, max_y),
            arrowprops=dict(arrowstyle="<->", color="#2980b9", lw=1.8), zorder=6
        )
        ax.text(
            6, (half_h + max_y) / 2,
            f"Top Margin:\n{p['top_margin']:.1f} mm",
            color="#2980b9", fontweight="bold", fontsize=10, va="center", zorder=6
        )

        # Left margin arrow
        ax.annotate(
            "", xy=(-half_w, 0), xytext=(min_x, 0),
            arrowprops=dict(arrowstyle="<->", color="#27ae60", lw=1.5), zorder=6
        )
        ax.text(
            (-half_w + min_x) / 2, -18,
            f"{p['left_margin']:.1f} mm",
            color="#27ae60", fontweight="bold", fontsize=9, ha="center", zorder=6
        )

        # Footprint label
        ax.text(
            0, (min_y + max_y) / 2,
            f"Footprint:\n{max_x - min_x:.1f} × {max_y - min_y:.1f} mm",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.9),
            zorder=7
        )

        # Information note box
        ax.text(
            -half_w + 5, -half_h + 8, p["note"],
            fontsize=8.5, color="#2c3e50",
            bbox=dict(boxstyle="square,pad=0.3", facecolor="#ecf0f1", edgecolor="none", alpha=0.9),
            zorder=7
        )

        ax.set_xlim(-130, 130)
        ax.set_ylim(-175, 175)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("u / X Coordinate (mm)", fontsize=11)
        ax.set_ylabel("v / Y Coordinate (mm)", fontsize=11)
        ax.set_title(p["title"], fontsize=12, fontweight="bold", pad=10)
        ax.grid(True, linestyle=":", alpha=0.4)

    # Plot Panel 3: Comparison between Old and Correct Run 9
    ax3 = panels[2]["ax"]
    pts_correct = panels[2]["pts_correct"]
    pts_old = panels[2]["pts_old"]

    rect3 = patches.Rectangle(
        (-half_w, -half_h), sheet_w, sheet_h,
        linewidth=2.0, edgecolor=sheet_edge, facecolor=sheet_color, zorder=1
    )
    ax3.add_patch(rect3)
    ax3.axhline(0, color="#95a5a6", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)
    ax3.axvline(0, color="#95a5a6", linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)

    # Fixture baseline line (v = -65 mm)
    ax3.axhline(-65.0, color="#8e44ad", linestyle="-.", linewidth=1.5, alpha=0.9, zorder=3, label="v = -65 mm (85 mm Fixture Datum)")

    # Plot old (erroneous) path in orange dashed
    ax3.plot(pts_old[:, 0], pts_old[:, 1], color="#e67e22", linestyle="--", linewidth=0.8, alpha=0.7, label="Old Config (Bottom Margin = 56.2 mm)", zorder=4)

    # Plot correct path in red solid
    ax3.plot(pts_correct[:, 0], pts_correct[:, 1], color="#d62728", linewidth=1.0, alpha=0.9, label="True Physical (Bottom Margin = 85.0 mm)", zorder=5)

    # Shift arrow showing +28.75 mm offset
    ax3.annotate(
        "", xy=(0, np.mean(pts_correct[:, 1])), xytext=(0, np.mean(pts_old[:, 1])),
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=2.5), zorder=6
    )
    ax3.text(
        8, 14,
        "Required Shift:\nΔv = +28.75 mm",
        color="#8e44ad", fontweight="bold", fontsize=10, va="center", zorder=6
    )

    # Margins for correct
    ax3.annotate(
        "", xy=(-45, -half_h), xytext=(-45, np.min(pts_correct[:, 1])),
        arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2.0), zorder=6
    )
    ax3.text(-40, -107.5, "True: 85 mm", color="#c0392b", fontweight="bold", fontsize=9, va="center")

    ax3.annotate(
        "", xy=(-75, -half_h), xytext=(-75, np.min(pts_old[:, 1])),
        arrowprops=dict(arrowstyle="<->", color="#e67e22", lw=1.8), zorder=6
    )
    ax3.text(-70, -121.8, "Old: 56.2 mm", color="#e67e22", fontweight="bold", fontsize=9, va="center")

    ax3.text(
        -half_w + 5, -half_h + 8, panels[2]["note"],
        fontsize=8.5, color="#2c3e50",
        bbox=dict(boxstyle="square,pad=0.3", facecolor="#ecf0f1", edgecolor="none", alpha=0.9),
        zorder=7
    )

    ax3.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
    ax3.set_xlim(-130, 130)
    ax3.set_ylim(-175, 175)
    ax3.set_aspect("equal", "box")
    ax3.set_xlabel("u / X Coordinate (mm)", fontsize=11)
    ax3.set_ylabel("v / Y Coordinate (mm)", fontsize=11)
    ax3.set_title(panels[2]["title"], fontsize=12, fontweight="bold", pad=10)
    ax3.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    out_png = OUT_DIR / "run6_run9_toolpaths_on_sheet.png"
    out_pdf = OUT_DIR / "run6_run9_toolpaths_on_sheet.pdf"
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"Saved corrected plots to {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
